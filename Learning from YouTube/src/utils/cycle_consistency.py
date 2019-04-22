"""
This is a script for finding the cycle-consistency / one to one alignment of a
sequence of embeddings

--------------------------------------------------------------------------------

This process is decribed in the paper ("Playing Hard Exploration Games By
Watching YouTube Videos" by Aytar et al.) here:

"Assume that we have two length-N sequences, V = {v_1, v_2,...v_n} and
W = {w_1, w_2, ..., w_n}. We also define the distance, dφ, as the Euclidean
distance in the associated embedding space, dφ(v_i , w_j) = ||φ(v_i) − φ(w_j)||2.
To evaluate cycle-consistency, we first select v_i ∈ V and determine its
nearest neighbor, w_j = argminw∈W dφ(v_i, w). We then repeat the process to find
the nearest neighbor of w_j , i.e. v_k = argminv∈V dφ(v, w_j). We say that v_i
is cycle-consistent if and only if |i − k| ≤ 1, and further define the
one-to-one alignment capacity, Pφ, of the embedding space φ as the percentage of
v ∈ V that are cycle-consistent. Figure 4(a) illustrates cycle-consistency in
two example embedding spaces. The same process can be extended to evaluate the
3-cycle-consistency of φ, Pφ_3, by requiring that vi remains cycle consistent
along both paths V → W → U → V and V → U → W → V, where U is a third sequence."

"""

def cycle_consistency(V, W):
"""
Loops through embedding sequence V and evaluates cycle consistency against
embedding sequence W.

Params:
    -V: Embedding (Tensor)
    -W: Embedding (Tensor)

Returns:
    -cycle consistency of embedding V
"""
