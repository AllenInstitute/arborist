"""
Created on Thu Aug 28 2026

CSR-backed graph storage used as the base class of "SkeletonGraph".

Implements the subset of the NetworkX Graph API that arborist and
neuron-proofreader use, over a pair of CSR arrays instead of nx's
dict-of-dicts. The motivation is memory: nx costs ~554 B/node, of which
~258 B/node is charged even with zero edges, which puts a cohort-scale
collection of skeletons at tens of gigabytes. The equivalent CSR structure
is ~42 B/node.

Three behaviours here are load-bearing and are not obvious from the API:

1.  Neighbour ordering is a constructor parameter. nx iterates adjacency in
    edge INSERTION order, and three consumers are sensitive to it:
    "component_to_zipped_swc" writes SWC rows in dfs_edges order,
    "cable_length(max_depth=...)" truncates mid-traversal, and
    "rooted_subgraph" assigns node IDs in discovery order. Sorted (canonical)
    CSR is the default because measurement showed no order-valued output
    changes, but "insertion_order=True" reproduces nx exactly and exists to
    bisect any case where that turns out to be wrong.

2.  Edges added after construction accumulate in an overlay that is served
    directly by "neighbors" rather than being folded in on the next read.
    Folding on read costs one O(E) rebuild per mutation for any caller that
    interleaves add_edge with adjacency reads, which the merge-proofreading
    accept loop does on every accept.

3.  Node IDs are a dense range, so "remove_nodes_from" cannot actually delete
    a node the way nx does. Removals are recorded in a mask, accumulated
    across calls, and applied by "relabel_nodes". Everything that reports on
    nodes honours the mask so that the observable behaviour matches nx.
"""

from array import array
from copy import deepcopy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _connected_components

import networkx as nx
import numpy as np

# Pending edges are folded in once they outnumber the materialized edges, so
# a bulk build costs O(log E) rebuilds instead of holding every edge in a
# Python list. Kept well above the ~2k accepts of a merge-proofreading run so
# that workload still pays zero rebuilds.
MIN_PENDING_BEFORE_FLUSH = 8192


class _DegreeView:
    """
    Degree lookup that supports both nx access forms.

    Of the degree call sites across the two repos, most use subscript syntax
    ("graph.degree[i]") and the rest use call syntax ("graph.degree(i)"), so
    both are supported and no consumer line has to change.

    Backed by a Python list rather than the numpy degree array because
    consumers read degrees one node at a time inside Python loops, where
    "int(arr[i])" on an ndarray costs a boxed numpy integer per access.
    """

    __slots__ = ("_degrees",)

    def __init__(self, degrees):
        self._degrees = degrees

    def __getitem__(self, i):
        return self._degrees[i]

    def __call__(self, i=None):
        return self if i is None else self._degrees[i]

    def __iter__(self):
        return iter(enumerate(self._degrees))

    def __len__(self):
        return len(self._degrees)


class _NodeView:
    """
    Node collection mimicking the parts of nx's NodeView that are used:
    iteration, membership, length and truthiness.
    """

    __slots__ = ("_ids", "_id_set")

    def __init__(self, ids):
        self._ids = ids
        self._id_set = None

    def __iter__(self):
        return iter(self._ids)

    def __len__(self):
        return len(self._ids)

    def __contains__(self, i):
        if isinstance(self._ids, range):
            return i in self._ids
        if self._id_set is None:
            self._id_set = set(self._ids)
        return i in self._id_set

    def __getitem__(self, i):
        return self._ids[i]

    def __repr__(self):
        return f"_NodeView({len(self._ids)} nodes)"


class CSRGraph:
    """
    Undirected graph stored as CSR arrays.

        _indptr   int64, (n + 1,)   row offsets into _indices
        _indices  int32, (2E,)      neighbour IDs, concatenated by row

    "_indices" is a zero-copy view of "_nbr", an array("i") that is the
    primary store because slicing it is faster than slicing an ndarray and
    converting, which is what "neighbors" does on every traversal step.
    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        node_spacing=1,
        verbose=False,
        insertion_order=False,
    ):
        """
        Instantiates a CSRGraph object.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors. Default is
            (1.0, 1.0, 1.0).
        node_spacing : float, optional
            Distance (in microns) between neighboring nodes. Default is 1μm.
        verbose : bool, optional
            Indication of whether to display progress bars. Default is False.
        insertion_order : bool, optional
            Indication of whether to order each node's adjacency by edge
            insertion time, reproducing nx's traversal order exactly, instead
            of by neighbour ID. Default is False.
        """
        self.anisotropy = np.array(anisotropy)
        self.component_id_to_swc_id = dict()
        self.kdtree = None
        self.node_spacing = node_spacing
        self.verbose = verbose
        self.insertion_order = insertion_order
        self._init_structure()

    def _init_structure(self):
        self._num_ids = 0
        self._indptr = np.zeros(1, dtype=np.int64)
        self._nbr = array("i")
        self._indices = np.zeros(0, dtype=np.int32)
        self._edge_time = None
        self._clock = 0

        self._pending = list()
        self._pending_adj = dict()
        self._dropped = None

        self._degrees = None
        self._degree_view = None
        self._indptr_list = None
        self._labels = None
        self._num_components = 0
        self._dsu = None
        self.rebuild_count = 0

    # --- Construction ---
    def build_from_edges(self, num_nodes, edges):
        """
        Builds the CSR structure from an edge list, discarding any existing
        structure.

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        edges : ArrayLike
            Array of node ID pairs with shape (E, 2).

        Returns
        -------
        CSRGraph
            Self, so that construction can be chained.
        """
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if len(edges):
            # nx.add_edge is a no-op on an existing edge and real SWCs repeat
            # edges. Keeping the first occurrence preserves insertion order.
            _, first = np.unique(
                np.sort(edges, axis=1), axis=0, return_index=True
            )
            edges = edges[np.sort(first)]

        self._pending = list()
        self._pending_adj = dict()
        self._dropped = None
        self._num_ids = int(num_nodes)
        self._clock = len(edges)
        self._materialize(edges, np.arange(len(edges), dtype=np.int64))
        return self

    def _materialize(self, edges, times):
        """
        Rebuilds the CSR arrays from an edge list. Does not clear the removal
        mask, which must survive until "relabel_nodes" applies it.
        """
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        stamps = np.concatenate([times, times])

        # nx creates nodes implicitly on add_edge and "rooted_subgraph"
        # depends on it, adding edges only and reading number_of_nodes() back
        # to allocate the next ID.
        if len(rows):
            self._num_ids = max(self._num_ids, int(rows.max()) + 1)

        # lexsort's last key is primary. Ordering within a row by insertion
        # time reproduces nx's adjacency order; ordering by neighbour ID gives
        # canonical CSR.
        second = stamps if self.insertion_order else cols
        order = np.lexsort((second, rows))

        self._set_indices(cols[order].astype(np.int32, copy=False))
        self._edge_time = (
            stamps[order].astype(np.int32, copy=False)
            if self.insertion_order else None
        )

        self._indptr = np.zeros(self._num_ids + 1, dtype=np.int64)
        if len(rows):
            counts = np.bincount(rows, minlength=self._num_ids)
            np.cumsum(counts, out=self._indptr[1:])

        self._pending = list()
        self._pending_adj = dict()
        self._drop_caches()
        self.rebuild_count += 1

    def _set_indices(self, values):
        """
        Stores neighbour IDs in an array("i") and exposes a zero-copy ndarray
        view of the same buffer.
        """
        self._nbr = array("i")
        self._nbr.frombytes(values.tobytes())
        self._indices = (
            np.frombuffer(self._nbr, dtype=np.int32) if len(self._nbr)
            else np.zeros(0, dtype=np.int32)
        )

    def _drop_caches(self):
        self._degrees = None
        self._degree_view = None
        self._indptr_list = None
        self._labels = None
        self._dsu = None

    def _flush(self):
        """
        Folds pending edges into the CSR structure.
        """
        if not self._pending:
            return
        old_edges = self._edge_list()
        new_edges = np.asarray(self._pending, dtype=np.int64).reshape(-1, 2)
        times = np.concatenate([
            self._edge_times(),
            np.arange(self._clock - len(new_edges), self._clock, dtype=np.int64)
        ])
        self._pending = list()
        self._materialize(
            np.concatenate([old_edges, new_edges]), times
        )

    def _rows(self):
        # Sized off indptr rather than _num_ids because add_edge and add_node
        # can grow the ID space ahead of the next rebuild.
        return np.repeat(
            np.arange(len(self._indptr) - 1), np.diff(self._indptr)
        )

    def _edge_list(self):
        """
        Returns the (E, 2) upper-triangle edge list, insertion-ordered when
        insertion times are being tracked.
        """
        rows = self._rows()
        keep = rows < self._indices
        edges = np.stack([rows[keep], self._indices[keep]], axis=1)
        if self._edge_time is None:
            return edges
        return edges[np.argsort(self._edge_time[keep], kind="stable")]

    def _edge_times(self):
        if self._edge_time is None:
            return np.arange(len(self._indices) // 2, dtype=np.int64)
        rows = self._rows()
        return np.sort(
            self._edge_time[rows < self._indices], kind="stable"
        )

    # --- Node Removal Mask ---
    def _drop_mask(self):
        if self._dropped is None:
            self._dropped = np.zeros(self._num_ids, dtype=bool)
        elif len(self._dropped) < self._num_ids:
            self._dropped = np.concatenate([
                self._dropped,
                np.zeros(self._num_ids - len(self._dropped), dtype=bool),
            ])
        return self._dropped

    def _surviving_ids(self):
        if self._dropped is None:
            return range(self._num_ids)
        return np.flatnonzero(~self._drop_mask()).tolist()

    # --- Structure ---
    @property
    def nodes(self):
        return _NodeView(self._surviving_ids())

    @property
    def edges(self):
        self._flush()
        return [(int(i), int(j)) for i, j in self._edge_list()]

    @property
    def degree(self):
        if self._degrees is None:
            degrees = np.diff(self._indptr).tolist()
            degrees.extend([0] * (self._num_ids - len(degrees)))
            # Degree needs no adjacency, so pending edges are counted rather
            # than materialized. This is what stops a run of accepts from
            # paying one rebuild per degree read.
            for i, j in self._pending:
                degrees[i] += 1
                degrees[j] += 1
            self._degrees = degrees
            self._degree_view = _DegreeView(degrees)
        return self._degree_view

    @property
    def _indptr_cached(self):
        """
        indptr as a Python list, because scalar ndarray indexing costs a boxed
        numpy integer per access and "neighbors" is on the hot path of every
        traversal in SkeletonGraph.
        """
        if self._indptr_list is None:
            self._indptr_list = self._indptr.tolist()
        return self._indptr_list

    def neighbors(self, i):
        """
        Returns the neighbours of node i, including pending edges.

        Parameters
        ----------
        i : int
            Node ID.

        Returns
        -------
        array.array
            Neighbouring node IDs.
        """
        indptr = self._indptr_cached
        # Nodes added since the last rebuild sit past the end of indptr. They
        # are not padded in, because padding on every add_node would make a
        # bulk build quadratic.
        base = (
            self._nbr[indptr[i]:indptr[i + 1]] if i + 1 < len(indptr)
            else array("i")
        )
        extra = self._pending_adj.get(i) if self._pending_adj else None
        return base if extra is None else base + array("i", extra)

    def number_of_nodes(self):
        if self._dropped is None:
            return self._num_ids
        return int((~self._drop_mask()).sum())

    def number_of_edges(self):
        return len(self._indices) // 2 + len(self._pending)

    def has_node(self, i):
        if not 0 <= i < self._num_ids:
            return False
        return self._dropped is None or not self._drop_mask()[i]

    def has_edge(self, i, j):
        return self._has_edge(int(i), int(j))

    def __len__(self):
        return self.number_of_nodes()

    def __iter__(self):
        return iter(self._surviving_ids())

    def __contains__(self, i):
        return self.has_node(i)

    def _csr(self):
        self._flush()
        indptr = self._indptr
        if len(indptr) < self._num_ids + 1:
            indptr = np.concatenate([
                indptr,
                np.full(
                    self._num_ids + 1 - len(indptr), indptr[-1],
                    dtype=np.int64,
                ),
            ])
        return csr_matrix(
            (
                np.ones(len(self._indices), dtype=np.int8),
                self._indices,
                indptr,
            ),
            shape=(self._num_ids, self._num_ids),
        )

    # --- Mutation ---
    def _has_edge(self, i, j):
        """
        Membership test against the CSR arrays plus the overlay, without
        materializing.
        """
        indptr = self._indptr_cached
        if i + 1 < len(indptr) and j in self._nbr[indptr[i]:indptr[i + 1]]:
            return True
        extra = self._pending_adj.get(i)
        return extra is not None and j in extra

    def add_edge(self, i, j):
        """
        Queues an edge to be added.

        The call is idempotent because nx.add_edge on an existing edge is a
        no-op; skipping that check makes degree and neighbors disagree with nx
        on any graph whose SWC file repeats an edge.

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID.
        """
        i, j = int(i), int(j)
        if self._has_edge(i, j):
            return
        self._pending.append((i, j))
        self._clock += 1
        grew = max(i, j) + 1 > self._num_ids
        self._num_ids = max(self._num_ids, i + 1, j + 1)

        self._pending_adj.setdefault(i, list()).append(j)
        self._pending_adj.setdefault(j, list()).append(i)

        if grew:
            # nx creates nodes implicitly. Only the node-sized caches are
            # invalidated; indptr is left short and "neighbors" ranges past its
            # end, because padding here would make a bulk build quadratic.
            self._degrees = None
            self._degree_view = None
            self._labels = None
            self._dsu = None
        else:
            if self._degrees is not None:
                self._degrees[i] += 1
                self._degrees[j] += 1
            if self._labels is not None:
                self._union(i, j)

        # Amortized growth: without this a bulk build holds every edge as a
        # Python tuple, which costs more than the nx graph being replaced.
        if len(self._pending) >= max(
            MIN_PENDING_BEFORE_FLUSH, len(self._indices) // 2
        ):
            self._flush()

    def add_edges_from(self, edges):
        for i, j in edges:
            self.add_edge(i, j)

    def add_node(self, i):
        i = int(i)
        if i < self._num_ids:
            if self._dropped is not None and self._drop_mask()[i]:
                self._drop_mask()[i] = False
                self._labels = None
                self._dsu = None
            return
        self._num_ids = i + 1
        self._degrees = None
        self._degree_view = None
        self._labels = None
        self._dsu = None

    def add_nodes_from(self, nodes):
        for i in nodes:
            self.add_node(i)

    def remove_nodes_from(self, nodes):
        """
        Removes nodes and their incident edges. Surviving node IDs are
        unchanged, matching nx, so "relabel_nodes" remains a separate step.

        Parameters
        ----------
        nodes : container
            Node IDs to remove.
        """
        idxs = np.fromiter(
            (int(i) for i in nodes), dtype=np.int64, count=-1
        )
        if not len(idxs):
            return
        self._flush()

        # Accumulated, not replaced: "trim_branches" removes nodes once per
        # trimmed proposal and relabels once at the end, so an overwriting
        # mask would resurrect everything but the last removal.
        dropped = self._drop_mask()
        dropped[idxs] = True

        edges = self._edge_list()
        times = self._edge_times()
        if len(edges):
            keep = ~(dropped[edges[:, 0]] | dropped[edges[:, 1]])
            edges, times = edges[keep], times[keep]
        self._materialize(edges, times)

    def _relabel_structure(self):
        """
        Compacts node IDs, applying any accumulated removals.

        Kept separate from the public "relabel_nodes" so that subclasses can
        reindex their node attribute arrays against the returned ordering.

        Returns
        -------
        Tuple[numpy.ndarray, dict]
            Old node IDs ordered by their new node ID, and the mapping from
            old node ID to new node ID.
        """
        self._flush()
        keep = np.ones(self._num_ids, dtype=bool)
        if self._dropped is not None:
            keep &= ~self._drop_mask()
        old_node_ids = np.flatnonzero(keep)
        old_to_new = np.full(self._num_ids, -1, dtype=np.int64)
        old_to_new[old_node_ids] = np.arange(len(old_node_ids))

        edges = self._edge_list()
        times = self._edge_times()
        edges = old_to_new[edges] if len(edges) else edges.reshape(-1, 2)

        self._dropped = None
        self._num_ids = len(old_node_ids)
        self._materialize(edges, times)
        return old_node_ids, {
            int(old): int(new)
            for old, new in zip(old_node_ids, range(self._num_ids))
        }

    # --- Connectivity ---
    def _component_labels(self):
        """
        Returns a per-node component label, and initializes a disjoint-set
        structure over those labels.

        The labels are computed once from the materialized CSR and incremental
        edges merge label classes in the disjoint-set structure rather than
        invalidating them, so connectivity queries survive a run of mutations
        without a rebuild.
        """
        if self._labels is None:
            _, self._labels = _connected_components(
                self._csr(), directed=False
            )
            num_labels = (
                int(self._labels.max()) + 1 if len(self._labels) else 0
            )
            self._dsu = list(range(num_labels))
            if self._dropped is None:
                self._num_components = num_labels
            else:
                surviving = self._labels[~self._drop_mask()]
                self._num_components = len(np.unique(surviving))
        return self._labels

    def _exact_labels(self):
        """
        Returns labels with pending edges folded in, so that equal labels mean
        the same component.
        """
        self._flush()
        return self._component_labels()

    def _find(self, x):
        dsu = self._dsu
        while dsu[x] != x:
            dsu[x] = dsu[dsu[x]]
            x = dsu[x]
        return x

    def _union(self, i, j):
        a = self._find(int(self._labels[i]))
        b = self._find(int(self._labels[j]))
        if a != b:
            self._dsu[b] = a
            self._num_components -= 1

    def connected_components(self):
        """
        Returns the node sets of every connected component.

        Returns
        -------
        List[set]
        """
        labels = self._exact_labels()
        if not len(labels):
            return list()
        if self._dropped is None:
            order = np.argsort(labels, kind="stable")
        else:
            survivors = np.flatnonzero(~self._drop_mask())
            order = survivors[
                np.argsort(labels[survivors], kind="stable")
            ]
        bounds = np.flatnonzero(np.diff(labels[order])) + 1
        return [set(c.tolist()) for c in np.split(order, bounds)]

    def number_connected_components(self):
        self._component_labels()
        return self._num_components

    def node_connected_component(self, i):
        labels = self._exact_labels()
        nodes = np.flatnonzero(labels == labels[i])
        if self._dropped is not None:
            nodes = nodes[~self._drop_mask()[nodes]]
        return set(nodes.tolist())

    def has_path(self, i, j):
        """
        Determines whether nodes i and j are connected.

        Compares component labels rather than running a traversal, because nx
        runs a breadth-first search per call and several consumers call this in
        a loop. Goes through the disjoint-set structure so that pending edges
        are honoured without a rebuild.

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID.

        Returns
        -------
        bool
        """
        labels = self._component_labels()
        return bool(
            self._find(int(labels[i])) == self._find(int(labels[j]))
        )

    def dfs_edges(self, source=None):
        """
        Iterates over edges in depth-first order.

        Replicates nx.dfs_edges, including its stack discipline and neighbour
        ordering, so that consumers which depend on traversal order produce
        identical output. scipy's depth_first_order is not a substitute: it
        orders neighbours by node ID and returns a node ordering rather than
        parent/child pairs.

        Parameters
        ----------
        source : int, optional
            Node to start from. Traverses every component if None. Default is
            None.

        Yields
        ------
        Tuple[int]
        """
        self._flush()
        indptr, indices = self._indptr_cached, self._nbr
        starts = self._surviving_ids() if source is None else [source]
        visited = set()
        for start in starts:
            if start in visited:
                continue
            visited.add(start)
            stack = [(start, indptr[start], indptr[start + 1])]
            while stack:
                parent, lo, hi = stack[-1]
                advanced = False
                while lo < hi:
                    child = indices[lo]
                    lo += 1
                    if child not in visited:
                        visited.add(child)
                        stack[-1] = (parent, lo, hi)
                        yield parent, child
                        stack.append(
                            (child, indptr[child], indptr[child + 1])
                        )
                        advanced = True
                        break
                if not advanced:
                    stack[-1] = (parent, lo, hi)
                    if lo >= hi:
                        stack.pop()

    def shortest_path(self, source, target):
        """
        Returns the shortest path between two nodes as a list of node IDs.

        Parameters
        ----------
        source : int
            Node ID to start from.
        target : int
            Node ID to end at.

        Returns
        -------
        List[int]

        Raises
        ------
        networkx.NetworkXNoPath
            If no path exists. Raised as the nx exception because consumers
            catch it by that name.
        """
        self._flush()
        indptr, indices = self._indptr_cached, self._nbr
        source, target = int(source), int(target)
        pred = np.full(self._num_ids, -2, dtype=np.int64)
        pred[source] = -1
        queue = [source]
        while queue and pred[target] == -2:
            next_queue = list()
            for i in queue:
                for k in range(indptr[i], indptr[i + 1]):
                    j = indices[k]
                    if pred[j] == -2:
                        pred[j] = i
                        next_queue.append(j)
            queue = next_queue
        if pred[target] == -2:
            raise nx.NetworkXNoPath(
                f"No path between {source} and {target}"
            )
        path = [target]
        while path[-1] != source:
            path.append(int(pred[path[-1]]))
        return path[::-1]

    def subgraph(self, nodes):
        """
        Returns the induced subgraph on the given nodes, with node IDs
        renumbered by ascending original ID.

        Parameters
        ----------
        nodes : container
            Node IDs to keep.

        Returns
        -------
        CSRGraph
        """
        idxs = np.array(sorted(nodes), dtype=np.int64)
        subgraph = self.__class__(
            anisotropy=self.anisotropy,
            node_spacing=self.node_spacing,
            verbose=self.verbose,
            insertion_order=self.insertion_order,
        )
        remap = np.full(self._num_ids, -1, dtype=np.int64)
        remap[idxs] = np.arange(len(idxs))

        self._flush()
        edges = self._edge_list()
        if len(edges):
            keep = (remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)
            edges = remap[edges[keep]]
        subgraph._num_ids = len(idxs)
        subgraph._materialize(
            edges.reshape(-1, 2), np.arange(len(edges), dtype=np.int64)
        )
        subgraph.component_id_to_swc_id = dict(self.component_id_to_swc_id)
        if hasattr(self, "copy_node_attrs"):
            self.copy_node_attrs(subgraph, idxs)
        return subgraph

    # --- Serialization ---
    def __getstate__(self):
        """
        Excludes the ndarray view of the neighbour buffer, which cannot be
        pickled or deep-copied without silently unlinking it from the
        array("i") it aliases.
        """
        state = dict(self.__dict__)
        state.pop("_indices", None)
        state["_nbr"] = self._nbr.tobytes()
        return state

    def __setstate__(self, state):
        nbr_bytes = state.pop("_nbr")
        self.__dict__.update(state)
        self._nbr = array("i")
        self._nbr.frombytes(nbr_bytes)
        self._indices = (
            np.frombuffer(self._nbr, dtype=np.int32) if len(self._nbr)
            else np.zeros(0, dtype=np.int32)
        )

    def __deepcopy__(self, memo):
        graph = self.__class__.__new__(self.__class__)
        memo[id(self)] = graph
        graph.__setstate__(deepcopy(self.__getstate__(), memo))
        return graph

    def nbytes(self):
        """
        Returns the number of bytes used by the structural arrays, excluding
        node attributes. "_indices" is a view of "_nbr", so only one is
        counted.
        """
        return (
            self._indptr.nbytes
            + self._nbr.buffer_info()[1] * self._nbr.itemsize
            + (0 if self._edge_time is None else self._edge_time.nbytes)
        )
