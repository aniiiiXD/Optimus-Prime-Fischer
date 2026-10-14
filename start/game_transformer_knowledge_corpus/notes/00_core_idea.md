# Core idea: from a board to a decision

## 1. Three different things people call a “game Transformer”

1. **Sequence model:** receives earlier moves and predicts the next move, as in OthelloGPT.
2. **Position model:** receives the current board and predicts a policy and/or value, as in modern chess engines.
3. **Decision model:** receives a state plus named candidate actions and scores each candidate, similar to the public Laya/Jev interface.

They share Transformer blocks, but their tokens, heads, losses, and inference loops differ.

## 2. The mathematical pipeline

Let a game position be (s), its legal actions be (A(s)), and the tokenized input be (x_1,\ldots,x_T). Token and position embeddings form

\[
h_i^{(0)} = E_{\text{token}}[x_i] + E_{\text{position}}[i] + E_{\text{type}}[t_i].
\]

Each Transformer block updates those vectors with causal self-attention and an MLP:

\[
Q=HW_Q,\quad K=HW_K,\quad V=HW_V,
\]

\[
\operatorname{Attention}(H)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V,
\]

\[
H' = H + \operatorname{Attention}(\operatorname{LN}(H)),\qquad
H^{+}=H'+\operatorname{MLP}(\operatorname{LN}(H')).
\]

The mask (M) is (-\infty) where information flow is forbidden. A causal mask blocks future moves; a legal-action mask blocks impossible output moves.

The policy head produces one logit (z_a) per action. Illegal actions are removed before normalization:

\[
\pi(a\mid s)=\frac{\exp(z_a)\,\mathbf 1[a\in A(s)]}{\sum_b \exp(z_b)\,\mathbf 1[b\in A(s)]}.
\]

The value head predicts the outcome from the current player's perspective:

\[
V(s)=\tanh(w_v^\top h_{\text{pool}})\in[-1,1].
\]

## 3. What training changes

At initialization the embedding, attention, MLP, policy, and value weights are random. For a target move (a^*) and result (r\in\{-1,0,1\}), a basic objective is

\[
\mathcal L = -\log \pi(a^*\mid s) + \lambda_v(V(s)-r)^2 - \lambda_H H(\pi).
\]

Backpropagation computes (\nabla_\theta\mathcal L), and an optimizer updates every parameter with (\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L). “Learning a strategy” means those repeated updates make useful actions receive larger logits in states where the strategy applies.

## 4. Pretraining, post-training, and inference

- **Pretraining:** learn broad game structure from many state/action examples, usually next-move imitation or legal-move prediction.
- **Post-training:** shift the policy toward winning using stronger targets, value learning, preference pairs, self-play, or reinforcement learning.
- **Inference:** freeze the weights, run a forward pass, mask illegal actions, and choose a move directly or through search.

The cleanest hackathon pipeline is: imitation pretraining → outcome/value post-training → search-assisted inference.

## 5. The Laya/Jev connection

The public decision interface is naturally mapped onto a game:

| Decision primitive | Game interpretation |
|---|---|
| `choice` | choose one candidate move |
| `score` | estimate an ordered property such as risk or move quality |
| `noul` | estimate a binary proposition such as “this move wins immediately” |

Laya's repository shows typed questions evaluated without autoregressive text generation. For our engine, use the same spirit but a game-native action vocabulary:

\[
(s, a_1,\ldots,a_k)\xrightarrow{\text{encoder}}(z_1,\ldots,z_k,V(s)).
\]

This is attractive when the hackathon reveals a new game: the adapter enumerates legal candidates, while the policy network's external contract remains stable.

## 6. Why OthelloGPT matters

OthelloGPT is trained only on move histories, not explicit 8×8 board images. Its `mingpt/model.py` builds token embeddings, learned positional embeddings, causal blocks, and a vocabulary projection. Probes then test whether internal activations contain the board state. The central lesson is that next-token prediction can force a model to internally reconstruct hidden game state.

That is scientifically interesting, but competitive play needs another question: among legal moves, which one improves the probability of winning? Legal-move accuracy and win rate must therefore be measured separately.
