// Router core for azroute: occupancy, delay-aware tree building, rip-up with undo,
// negotiated-congestion initial routing. Exposed to Python via pybind11.
//
// Vertex ids match m3d.grid: vid = (z*H + y)*W + x.
// Legality is enforced on vertices only: every tree edge joins two vertices of
// the same net, so vertex-exclusive ownership implies edge-exclusive ownership.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <tuple>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

struct ModeParams {
    std::vector<double> layer_mult;  // per-layer multiplier on in-layer edge delay (search cost only)
    double via_mult = 1.0;           // multiplier on via delay (search cost only)
    double price_weight = 0.0;       // secondary cost: price_weight * price[v]
    double reserve_weight = 0.0;     // secondary cost for entering a reserved pin column
    double wire_weight = 0.0;        // secondary cost per newly added vertex
    bool zero_seed = false;          // seed every tree vertex with cost 0 (min-wire growth)
    bool far_sink_first = false;     // attach sinks one at a time, farthest first
};

struct Box {
    int x0, y0, x1, y1;  // inclusive
};

struct Route {
    bool routed = false;
    std::vector<int32_t> verts;
    std::vector<std::pair<int32_t, int32_t>> edges;  // (parent, child)
    int64_t delay = 0;                               // sum of driver->sink delays
};

class Router {
public:
    Router(int W, int H, int L, std::vector<int> layer_delay, int via_delay,
           const std::vector<int32_t>& pin_vids, const std::vector<int32_t>& pin_nets,
           const std::vector<int32_t>& drivers, const std::vector<std::vector<int32_t>>& sinks)
        : W_(W), H_(H), L_(L), WH_((int64_t)W * H), V_((int64_t)W * H * L),
          ld_(std::move(layer_delay)), via_(via_delay), drivers_(drivers), sinks_(sinks) {
        if ((int)ld_.size() != L_) throw std::invalid_argument("layer_delay size != layers");
        pin_owner_.assign(V_, -1);
        for (size_t i = 0; i < pin_vids.size(); ++i) pin_owner_[pin_vids[i]] = pin_nets[i];
        owner_.assign(V_, -1);
        routes_.resize(drivers_.size());
        dist_.assign(V_, 0.0);
        parent_.assign(V_, -1);
        dstamp_.assign(V_, 0);
        cstamp_.assign(V_, 0);
        tstamp_.assign(V_, 0);
        target_stamp_.assign(V_, 0);
        tdist_.assign(V_, 0);
        tcost_.assign(V_, 0.0);
        col_stamp_.assign(WH_, 0);
        min_ld_ = *std::min_element(ld_.begin(), ld_.end());
        // per (z, tz): best line over horizontal layer z' -> h = a*dxy + b
        hline_.assign((size_t)L_ * L_, {});
        for (int z = 0; z < L_; ++z)
            for (int tz = 0; tz < L_; ++tz)
                for (int zp = 0; zp < L_; ++zp)
                    hline_[(size_t)z * L_ + tz].push_back(
                        {(double)ld_[zp], (double)via_ * (std::abs(z - zp) + std::abs(zp - tz))});
    }

    int num_nets() const { return (int)drivers_.size(); }
    int64_t num_vertices() const { return V_; }

    // ------------------------------------------------------------------ queries
    bool is_routed(int net) const { return routes_.at(net).routed; }
    int64_t net_delay(int net) const { return routes_.at(net).delay; }
    int64_t total_delay() const {
        int64_t s = 0;
        for (auto& r : routes_) s += r.delay;
        return s;
    }
    bool all_routed() const {
        for (auto& r : routes_)
            if (!r.routed) return false;
        return true;
    }
    std::vector<int32_t> route_vertices(int net) const { return routes_.at(net).verts; }
    std::vector<std::pair<int32_t, int32_t>> route_edges(int net) const { return routes_.at(net).edges; }

    py::array_t<int32_t> owner_array() const {
        py::array_t<int32_t> a(V_);
        std::copy(owner_.begin(), owner_.end(), a.mutable_data());
        return a;
    }
    py::array_t<int32_t> pin_owner_array() const {
        py::array_t<int32_t> a(V_);
        std::copy(pin_owner_.begin(), pin_owner_.end(), a.mutable_data());
        return a;
    }

    // Owner and pin-owner of every vertex in the box [x0,x1]x[y0,y1] (all layers), shape (L, h, w).
    py::tuple crop(int x0, int y0, int x1, int y1) const {
        x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(W_ - 1, x1); y1 = std::min(H_ - 1, y1);
        const int w = x1 - x0 + 1, h = y1 - y0 + 1;
        py::array_t<int32_t> own({L_, h, w}), pin({L_, h, w});
        int32_t* po = own.mutable_data();
        int32_t* pp = pin.mutable_data();
        for (int z = 0; z < L_; ++z)
            for (int y = 0; y < h; ++y) {
                int64_t base = ((int64_t)z * H_ + (y0 + y)) * W_ + x0;
                std::copy(owner_.begin() + base, owner_.begin() + base + w, po);
                std::copy(pin_owner_.begin() + base, pin_owner_.begin() + base + w, pp);
                po += w; pp += w;
            }
        return py::make_tuple(own, pin);
    }

    // Minimum delay of the net ignoring every other net's routes (other pins still block).
    int64_t ideal_delay(int net) {
        Route r;
        ModeParams m = default_mode();
        Box b{0, 0, W_ - 1, H_ - 1};
        if (!build(net, m, b, nullptr, {}, BlockNone{}, NoCong{}, r)) return -1;
        return r.delay;
    }

    // Vertices of the net's ideal tree within `box`, ignoring every other net's routes
    // (other pins still block). Empty if unroutable.
    std::vector<int32_t> ideal_tree(int net, std::tuple<int, int, int, int> box) {
        Route r;
        ModeParams m = default_mode();
        if (!build(net, m, to_box(box), nullptr, {}, BlockNone{}, NoCong{}, r)) return {};
        return r.verts;
    }

    // Nets whose current routes occupy the net's ideal (unobstructed) tree within `box`.
    std::vector<int32_t> blockers(int net, std::tuple<int, int, int, int> box) {
        Route r;
        ModeParams m = default_mode();
        Box b = to_box(box);
        std::vector<int32_t> out;
        if (!build(net, m, b, nullptr, {}, BlockNone{}, NoCong{}, r)) return out;
        for (int32_t v : r.verts) {
            int o = owner_[v];
            if (o >= 0 && o != net && std::find(out.begin(), out.end(), o) == out.end()) out.push_back(o);
        }
        return out;
    }

    // ------------------------------------------------------------------ edits (undoable)
    // Route an unrouted net under hard capacity. Returns false (no change) if blocked.
    bool route_net(int net, const ModeParams& m, std::tuple<int, int, int, int> box,
                   py::object price, const std::vector<int32_t>& reserve_nets) {
        Route& cur = routes_.at(net);
        if (cur.routed) throw std::runtime_error("route_net: net already routed; rip it up first");
        check_mode(m);
        const float* pr = nullptr;
        py::array_t<float, py::array::c_style | py::array::forcecast> parr;
        if (!price.is_none()) {
            parr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(price);
            if (!parr || parr.size() != V_) throw std::invalid_argument("price must have one value per vertex");
            pr = parr.data();
        }
        Route r;
        bool ok;
        {
            py::gil_scoped_release release;
            ok = build(net, m, to_box(box), pr, reserve_nets, BlockOwned{this, net}, NoCong{}, r);
        }
        if (!ok) return false;
        undo_.push_back({net, std::move(cur)});
        install(net, std::move(r));
        return true;
    }

    // Install a given tree (unordered vertex-id edges) for an unrouted net. Validates that it
    // is a tree rooted at the driver that reaches every sink without touching another
    // net's vertices or pins; recomputes the exact delay. Undoable.
    void set_route(int net, const std::vector<std::pair<int32_t, int32_t>>& edges) {
        Route& cur = routes_.at(net);
        if (cur.routed) throw std::runtime_error("set_route: net already routed; rip it up first");
        std::vector<std::vector<std::pair<int32_t, int>>> adj;
        std::vector<int32_t> ids;
        std::unordered_map<int32_t, int> pos;
        auto idx = [&](int32_t v) -> int {
            auto it = pos.find(v);
            if (it != pos.end()) return it->second;
            ids.push_back(v);
            adj.emplace_back();
            return pos[v] = (int)ids.size() - 1;
        };
        for (auto& e : edges) {
            int32_t a = e.first, b = e.second;
            if (a < 0 || b < 0 || a >= V_ || b >= V_) throw std::invalid_argument("set_route: vertex out of range");
            int64_t d = std::llabs((int64_t)a - b);
            int w;
            if (d == WH_) w = via_;
            else if (d == 1 || d == W_) {
                int ax, ay, az, bx, by, bz;
                coord(a, ax, ay, az); coord(b, bx, by, bz);
                if (az != bz || std::abs(ax - bx) + std::abs(ay - by) != 1) throw std::invalid_argument("set_route: illegal edge");
                w = ld_[az];
            } else throw std::invalid_argument("set_route: illegal edge");
            int ia = idx(a), ib = idx(b);
            adj[ia].push_back({b, w});
            adj[ib].push_back({a, w});
        }
        int32_t driver = drivers_[net];
        if (ids.empty()) idx(driver);
        if (!pos.count(driver)) throw std::invalid_argument("set_route: tree misses driver");
        if (edges.size() + 1 != ids.size()) throw std::invalid_argument("set_route: not a tree");
        std::vector<int64_t> td(ids.size(), -1);
        std::vector<int> stack{idx(driver)};
        td[stack[0]] = 0;
        Route r;
        while (!stack.empty()) {
            int i = stack.back(); stack.pop_back();
            int32_t v = ids[i];
            int o = owner_[v], po = pin_owner_[v];
            if ((o >= 0 && o != net) || (po >= 0 && po != net)) throw std::invalid_argument("set_route: conflict");
            r.verts.push_back(v);
            for (auto& nb : adj[i]) {
                int j = idx(nb.first);
                if (td[j] >= 0) continue;
                td[j] = td[i] + nb.second;
                r.edges.push_back({v, nb.first});
                stack.push_back(j);
            }
        }
        if (r.verts.size() != ids.size()) throw std::invalid_argument("set_route: disconnected");
        int64_t total = 0;
        for (int32_t s : sinks_[net]) {
            auto it = pos.find(s);
            if (it == pos.end()) throw std::invalid_argument("set_route: sink not reached");
            total += td[it->second];
        }
        r.delay = total;
        r.routed = true;
        undo_.push_back({net, std::move(cur)});
        install(net, std::move(r));
    }

    // set_route for a tree known to be valid for this net, returning false (no change)
    // instead of throwing when one of its vertices is now owned by another net.
    bool try_set_route(int net, const std::vector<std::pair<int32_t, int32_t>>& edges) {
        auto taken = [&](int32_t v) { int o = owner_[v]; return o >= 0 && o != net; };
        if (taken(drivers_[net])) return false;
        for (auto& e : edges)
            if (taken(e.first) || taken(e.second)) return false;
        set_route(net, edges);
        return true;
    }

    void rip_up(int net) {
        Route& cur = routes_.at(net);
        if (!cur.routed) return;
        for (int32_t v : cur.verts)
            if (owner_[v] == net) owner_[v] = -1;
        undo_.push_back({net, std::move(cur)});
        routes_[net] = Route{};
    }

    size_t checkpoint() const { return undo_.size(); }

    void rollback(size_t cp) {
        while (undo_.size() > cp) {
            UndoEntry e = std::move(undo_.back());
            undo_.pop_back();
            Route& cur = routes_[e.net];
            if (cur.routed)
                for (int32_t v : cur.verts)
                    if (owner_[v] == e.net) owner_[v] = -1;
            routes_[e.net] = Route{};
            if (e.old.routed) install(e.net, std::move(e.old));
        }
    }

    void commit() { undo_.clear(); }

    // ------------------------------------------------------------------ initial routing
    // Negotiated congestion over all nets from scratch. On success the routing is
    // installed (legal) and true is returned; on failure nothing is installed.
    bool negotiate(const std::vector<int32_t>& order, const ModeParams& m, int max_iters,
                   double pres0, double pres_mult, double hist_fac) {
        check_mode(m);
        py::gil_scoped_release release;
        const int N = num_nets();
        for (int n = 0; n < N; ++n) {
            for (int32_t v : routes_[n].verts)
                if (owner_[v] == n) owner_[v] = -1;
            routes_[n] = Route{};
        }
        undo_.clear();
        std::vector<uint16_t> occ(V_, 0);
        std::vector<float> hist(V_, 0.f);
        std::vector<Route> soft(N);
        Box full{0, 0, W_ - 1, H_ - 1};
        double pres = pres0;
        auto add = [&](int n) { for (int32_t v : soft[n].verts) ++occ[v]; };
        auto remove = [&](int n) { for (int32_t v : soft[n].verts) --occ[v]; };
        for (int32_t n : order) {
            if (!build(n, m, full, nullptr, {}, BlockNone{}, SoftCong{&occ, &hist, pres}, soft[n])) { last_overuse_ = -1; return false; }
            add(n);
        }
        bool legal = false;
        last_iters_ = 0;
        for (int it = 0; it < max_iters; ++it) {
            last_iters_ = it + 1;
            std::vector<char> affected(N, 0);
            bool over = false;
            for (int n = 0; n < N; ++n)
                for (int32_t v : soft[n].verts)
                    if (occ[v] > 1) { affected[n] = 1; over = true; break; }
            if (!over) { legal = true; break; }
            for (int64_t v = 0; v < V_; ++v)
                if (occ[v] > 1) hist[v] += (float)(hist_fac * (occ[v] - 1));
            pres = std::min(pres * pres_mult, 1e6);
            for (int32_t n : order) {
                if (!affected[n]) continue;
                remove(n);
                Route r;
                if (!build(n, m, full, nullptr, {}, BlockNone{}, SoftCong{&occ, &hist, pres}, r)) { last_overuse_ = -1; return false; }
                soft[n] = std::move(r);
                add(n);
            }
        }
        if (!legal) {
            legal = true;
            for (int64_t v = 0; v < V_ && legal; ++v)
                if (occ[v] > 1) legal = false;
        }
        last_overuse_ = 0;
        for (int64_t v = 0; v < V_; ++v)
            if (occ[v] > 1) ++last_overuse_;
        if (!legal) return false;
        for (int n = 0; n < N; ++n) install(n, std::move(soft[n]));
        return true;
    }

    std::pair<int, int64_t> negotiate_stats() const { return {last_iters_, last_overuse_}; }

    static ModeParams default_mode_for(int L) {
        ModeParams m;
        m.layer_mult.assign(L, 1.0);
        return m;
    }

private:
    struct UndoEntry {
        int net;
        Route old;
    };
    struct BlockNone {
        bool operator()(int64_t) const { return false; }
    };
    struct BlockOwned {
        const Router* r;
        int net;
        bool operator()(int64_t v) const { int o = r->owner_[v]; return o >= 0 && o != net; }
    };
    struct NoCong {
        double operator()(int64_t) const { return 1.0; }
    };
    struct SoftCong {
        const std::vector<uint16_t>* occ;
        const std::vector<float>* hist;
        double pres;
        double operator()(int64_t v) const { return (1.0 + (*hist)[v]) * (1.0 + pres * (*occ)[v]); }
    };

    ModeParams default_mode() const { return default_mode_for(L_); }

    void check_mode(const ModeParams& m) const {
        if ((int)m.layer_mult.size() != L_) throw std::invalid_argument("mode.layer_mult needs one entry per layer");
        for (double x : m.layer_mult)
            if (x < 1.0) throw std::invalid_argument("mode.layer_mult entries must be >= 1 (keeps A* admissible)");
        if (m.via_mult < 1.0 || m.price_weight < 0 || m.reserve_weight < 0 || m.wire_weight < 0)
            throw std::invalid_argument("mode multipliers must be >= 1 and secondary weights >= 0");
    }

    Box to_box(const std::tuple<int, int, int, int>& t) const {
        Box b{std::max(0, std::get<0>(t)), std::max(0, std::get<1>(t)),
              std::min(W_ - 1, std::get<2>(t)), std::min(H_ - 1, std::get<3>(t))};
        return b;
    }

    void install(int net, Route&& r) {
        for (int32_t v : r.verts) owner_[v] = net;
        routes_[net] = std::move(r);
    }

    inline void coord(int64_t v, int& x, int& y, int& z) const {
        z = (int)(v / WH_);
        int64_t r = v - (int64_t)z * WH_;
        y = (int)(r / W_);
        x = (int)(r - (int64_t)y * W_);
    }

    // Exact obstacle-free true-delay distance between two vertices: consistent A* heuristic
    // for any search cost that is >= the true delay.
    inline double hdist(int x, int y, int z, int tx, int ty, int tz) const {
        int dxy = std::abs(x - tx) + std::abs(y - ty);
        if (dxy == 0) return (double)via_ * std::abs(z - tz);
        double best = std::numeric_limits<double>::infinity();
        for (auto& ln : hline_[(size_t)z * L_ + tz]) best = std::min(best, ln.first * dxy + ln.second);
        return best;
    }

    // Grow a tree for `net`. Sinks are attached by multi-source A* from the current
    // tree; each tree vertex is seeded with its search cost from the driver (or 0
    // when zero_seed), so DELAY mode yields a shortest-path tree. The search never
    // enters an existing tree vertex, so the result is always a tree.
    template <class Blocked, class Cong>
    bool build(int net, const ModeParams& m, const Box& box, const float* price,
               const std::vector<int32_t>& reserve_nets, Blocked blocked, Cong cong, Route& out) {
        out = Route{};
        const int32_t driver = drivers_[net];
        const auto& sinks = sinks_[net];
        const uint32_t T = ++tree_gen_;
        // reserved pin columns: pins of the given nets other than this one
        const uint32_t RS = ++col_gen_;
        bool any_reserve = m.reserve_weight > 0 && !reserve_nets.empty();
        if (any_reserve) {
            for (int32_t rn : reserve_nets) {
                if (rn == net) continue;
                col_stamp_[drivers_[rn] % WH_] = RS;
                for (int32_t s : sinks_[rn]) col_stamp_[s % WH_] = RS;
            }
        }
        out.verts.push_back(driver);
        tstamp_[driver] = T;
        tdist_[driver] = 0;
        tcost_[driver] = 0.0;
        std::vector<int32_t> remaining(sinks.begin(), sinks.end());
        int dx_, dy_, dz_;
        coord(driver, dx_, dy_, dz_);
        if (m.far_sink_first) {
            std::sort(remaining.begin(), remaining.end(), [&](int32_t a, int32_t b) {
                int ax, ay, az, bx, by, bz;
                coord(a, ax, ay, az);
                coord(b, bx, by, bz);
                double ha = hdist(dx_, dy_, dz_, ax, ay, az), hb = hdist(dx_, dy_, dz_, bx, by, bz);
                return ha != hb ? ha > hb : a < b;
            });
        }
        using QE = std::pair<double, int64_t>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        std::vector<std::array<int, 3>> tgt;
        auto edge_cost = [&](int64_t v, int z, bool via) -> std::pair<int, double> {
            int w = via ? via_ : ld_[z];
            double c = w * (via ? m.via_mult : m.layer_mult[z]);
            double cg = cong(v);
            if (cg != 1.0) c *= cg;
            if (price && m.price_weight != 0.0) c += m.price_weight * price[v];
            if (any_reserve && col_stamp_[v % WH_] == RS) c += m.reserve_weight;
            c += m.wire_weight;
            return {w, c};
        };
        while (!remaining.empty()) {
            const uint32_t S = ++search_gen_;
            tgt.clear();
            size_t ntg = m.far_sink_first ? 1 : remaining.size();
            for (size_t i = 0; i < ntg; ++i) {
                int tx, ty, tz;
                coord(remaining[i], tx, ty, tz);
                tgt.push_back({tx, ty, tz});
                target_stamp_[remaining[i]] = S;
            }
            const bool use_h = tgt.size() <= 8;
            auto h = [&](int x, int y, int z) -> double {
                if (!use_h) return 0.0;
                double b = std::numeric_limits<double>::infinity();
                for (auto& t : tgt) b = std::min(b, hdist(x, y, z, t[0], t[1], t[2]));
                return b;
            };
            pq = decltype(pq)();
            for (int32_t s : out.verts) {
                double g = m.zero_seed ? 0.0 : tcost_[s];
                dist_[s] = g;
                dstamp_[s] = S;
                parent_[s] = -1;
                int x, y, z;
                coord(s, x, y, z);
                pq.push({g + h(x, y, z), s});
            }
            int64_t found = -1;
            while (!pq.empty()) {
                const int64_t u = pq.top().second;
                pq.pop();
                if (cstamp_[u] == S) continue;
                cstamp_[u] = S;
                if (target_stamp_[u] == S && tstamp_[u] != T) { found = u; break; }
                int x, y, z;
                coord(u, x, y, z);
                const double du = dist_[u];
                auto relax = [&](int64_t v, int vx, int vy, int vz, bool via) {
                    if (tstamp_[v] == T) return;          // never re-enter own tree (keeps it acyclic)
                    if (cstamp_[v] == S) return;
                    int po = pin_owner_[v];
                    if (po >= 0 && po != net) return;     // never through another net's pin
                    if (blocked(v)) return;
                    auto wc = edge_cost(v, via ? vz : z, via);
                    double nd = du + wc.second;
                    if (dstamp_[v] != S || nd < dist_[v]) {
                        dstamp_[v] = S;
                        dist_[v] = nd;
                        parent_[v] = u;
                        pq.push({nd + h(vx, vy, vz), v});
                    }
                };
                if (x + 1 <= box.x1) relax(u + 1, x + 1, y, z, false);
                if (x - 1 >= box.x0) relax(u - 1, x - 1, y, z, false);
                if (y + 1 <= box.y1) relax(u + W_, x, y + 1, z, false);
                if (y - 1 >= box.y0) relax(u - W_, x, y - 1, z, false);
                if (z + 1 < L_) relax(u + WH_, x, y, z + 1, true);
                if (z - 1 >= 0) relax(u - WH_, x, y, z - 1, true);
            }
            if (found < 0) return false;
            // walk back to the tree, then add the path outward so tdist/tcost are exact
            path_.clear();
            int64_t cur = found;
            while (tstamp_[cur] != T) {
                path_.push_back(cur);
                cur = parent_[cur];
            }
            int64_t prev = cur;
            for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
                int64_t v = *it;
                int px, py, pz, vx, vy, vz;
                coord(prev, px, py, pz);
                coord(v, vx, vy, vz);
                bool via = pz != vz;
                auto wc = edge_cost(v, via ? vz : pz, via);
                tdist_[v] = tdist_[prev] + wc.first;
                tcost_[v] = tcost_[prev] + wc.second;
                tstamp_[v] = T;
                out.verts.push_back((int32_t)v);
                out.edges.push_back({(int32_t)prev, (int32_t)v});
                prev = v;
            }
            remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                                           [&](int32_t s) { return tstamp_[s] == T; }),
                            remaining.end());
        }
        int64_t total = 0;
        for (int32_t s : sinks) total += tdist_[s];
        out.delay = total;
        out.routed = true;
        return true;
    }

    int W_, H_, L_;
    int64_t WH_, V_;
    std::vector<int> ld_;
    int via_;
    int min_ld_;
    std::vector<int32_t> drivers_;
    std::vector<std::vector<int32_t>> sinks_;
    std::vector<int32_t> pin_owner_, owner_;
    std::vector<Route> routes_;
    std::vector<UndoEntry> undo_;
    std::vector<std::vector<std::pair<double, double>>> hline_;
    // scratch (stamp-indexed so nothing is cleared per search)
    std::vector<double> dist_, tcost_;
    std::vector<int64_t> parent_, path_;
    std::vector<int64_t> tdist_;
    std::vector<uint32_t> dstamp_, cstamp_, tstamp_, target_stamp_, col_stamp_;
    uint32_t search_gen_ = 0, tree_gen_ = 0, col_gen_ = 0;
    int last_iters_ = 0;
    int64_t last_overuse_ = 0;
};

PYBIND11_MODULE(_core, mod) {
    mod.doc() = "azroute router core";
    py::class_<ModeParams>(mod, "ModeParams")
        .def(py::init<>())
        .def_readwrite("layer_mult", &ModeParams::layer_mult)
        .def_readwrite("via_mult", &ModeParams::via_mult)
        .def_readwrite("price_weight", &ModeParams::price_weight)
        .def_readwrite("reserve_weight", &ModeParams::reserve_weight)
        .def_readwrite("wire_weight", &ModeParams::wire_weight)
        .def_readwrite("zero_seed", &ModeParams::zero_seed)
        .def_readwrite("far_sink_first", &ModeParams::far_sink_first);
    py::class_<Router>(mod, "Router")
        .def(py::init<int, int, int, std::vector<int>, int, const std::vector<int32_t>&,
                      const std::vector<int32_t>&, const std::vector<int32_t>&,
                      const std::vector<std::vector<int32_t>>&>())
        .def("num_nets", &Router::num_nets)
        .def("num_vertices", &Router::num_vertices)
        .def("is_routed", &Router::is_routed)
        .def("net_delay", &Router::net_delay)
        .def("total_delay", &Router::total_delay)
        .def("all_routed", &Router::all_routed)
        .def("route_vertices", &Router::route_vertices)
        .def("route_edges", &Router::route_edges)
        .def("owner_array", &Router::owner_array)
        .def("pin_owner_array", &Router::pin_owner_array)
        .def("ideal_delay", &Router::ideal_delay)
        .def("crop", &Router::crop)
        .def("blockers", &Router::blockers)
        .def("ideal_tree", &Router::ideal_tree)
        .def("route_net", &Router::route_net, py::arg("net"), py::arg("mode"), py::arg("box"),
             py::arg("price") = py::none(), py::arg("reserve_nets") = std::vector<int32_t>{})
        .def("rip_up", &Router::rip_up)
        .def("set_route", &Router::set_route)
        .def("try_set_route", &Router::try_set_route)
        .def("checkpoint", &Router::checkpoint)
        .def("rollback", &Router::rollback)
        .def("commit", &Router::commit)
        .def("negotiate", &Router::negotiate)
        .def("negotiate_stats", &Router::negotiate_stats);
}
