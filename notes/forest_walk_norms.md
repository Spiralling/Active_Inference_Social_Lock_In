# Forest Walk — Norm Evolution Model Redesign

**Date**: 2026-05-09
**Purpose**: Stress-test a multi-agent norm-evolution model design with experts spanning cultural evolution, norm theory, MARL, and active inference.
**Verification stage**: 1–2 (exploratory; for early frame-finding, not publication-ready).

---

## The setting

We're walking through old-growth coastal forest in the Pacific Northwest, late morning. Mist still pools in the hollows, and the path is moss-soft underfoot. The cedars are wet and aromatic; somewhere a varied thrush is calling. Five companions walk with me along a trail that climbs gently toward a clearing we can see two miles ahead through the trees.

**Companions:**

1. **Cristina Bicchieri** (Penn) — author of *The Grammar of Society* and *Norms in the Wild*. Holds that norms are *conditional preferences* held in equilibrium by mutual expectations: people follow norm $N$ when they believe enough others follow it (empirical expectation) and that enough others think they should follow it (normative expectation). Skeptical of cognitivist framings that locate norms inside individual heads.

2. **Joseph Henrich** (Harvard) — *The Secret of Our Success*, *The WEIRDest People*. Norms as cultural inheritance transmitted through prestige bias and conformist transmission. Variation between societies is variation in transmission outcomes, not in underlying preferences. Insists humans are *cultural learners first*, optimizers second.

3. **Joel Leibo** (DeepMind) — sequential social dilemmas, Melting Pot. Norms as equilibrium-selection devices in repeated multi-agent games. Demands formal Markov-game specifications, action-channel observation, and reputation as explicit state.

4. **David Hyland** — co-author of *On the Variational Costs of Changing Our Minds*. The $\lambda$-conservatism formalism. Wants to know exactly which mathematical object $\lambda$ tempers in the new model.

5. **Maxwell Ramstead** — cultural active inference, *Epistemic Communities Under Active Inference* (Albarracin, Demekas, Ramstead, Heins 2022). Sees norms as shared generative models — communities are Markov blankets within which agents' priors are correlated.

---

## §1 — The opening claim (along the lower trail)

**Me**: I want to model norm evolution where each agent encounters one micro-situation — say, a beggar approaches them on the street — and has five action choices: give money, give food, ignore, scold, redirect to services. Norms tell agents which action to take, neighbours observe the action, and reputation feeds back as payoff. The variational cost of mind-change makes some communities slow to revise, even when reputation payoffs shift. I'm calling the norms *attentional priors*.

**Bicchieri** stops at a fern stand, picks up a stone, and turns it over.

> "Stop right there. *Attentional priors* sounds like you've already lost the thread. A norm isn't a thing inside one agent's head. It's a *coordination equilibrium* sustained by mutual expectations. Your beggar example is perfect — let me run it.
>
> "An agent gives money to the beggar. Why? Three possibilities. One: she has an *internalized value* of generosity. Two: she has a *personal preference* for giving, unconditional on others. Three: she expects that *enough others in her community* both give *and* think she should give. Only the third is a norm in my sense. The first two are values or preferences.
>
> "Drop the third condition — make her believe nobody else gives, and nobody thinks she should — and the behaviour vanishes. That's the diagnostic. Norms are *contingent on beliefs about others*."

**Henrich** has been listening, walking with his hands in his pockets. He breaks in:

> "Cristina, I think you're right that norms are not just internal values, but you're missing the cumulative-cultural piece. The reason your agent's beliefs about others have the *content* they have — give, don't give, give to whom — is that those beliefs were *transmitted to her*, and not by random sampling. She copied prestigious individuals. She conformed to majorities at moments of uncertainty. The norm isn't sitting on top of conditional preferences as a meta-fact about expectations; it's the *output* of generations of biased social learning.
>
> "If you want the beggar example: in WEIRD societies, giving to street beggars is increasingly rare not because everyone's expectations of giving have shifted in lockstep — they shifted because *prestigious models* (effective-altruism advocates, social workers, certain politicians) made redirect-to-services the high-status response. Reputation didn't shift because the norm shifted; the norm shifted because reputation followed prestigious behaviour."

**Leibo** is walking faster than the rest of us. He turns and walks backward briefly:

> "Both of you are arguing about *what is being transmitted* before establishing *what game is being played*. Until I know the payoff structure, I don't know what norm is even *for*.
>
> "Here's what I want. The beggar interaction repeated $T$ times across a population of $N$ agents on graph $G$. Each agent plays a stage game with action set $\mathcal{A} = \{$ give-money, give-food, ignore, scold, redirect $\}$. Stage payoffs come from two channels: (a) some intrinsic payoff structure — maybe scold has a small private benefit because you don't have to engage; redirect has a moderate cost because you have to know the services; (b) a reputation payoff — your neighbours observe your action, judge it against *their* prior over what's appropriate, and assign you reputation that aggregates into resources next round.
>
> "Now you have a *Markov game*. Norms are equilibrium selections in this game. Multiple equilibria exist. Which one a community lands on is a coordination problem. *That's the level of analysis*. Cristina's expectations are the local-stability mechanism; Joe's transmission is the dynamics that select between basins."

**Bicchieri**: "I don't disagree that norms must be supported by some incentive structure. But the *content* of the norm — what counts as appropriate — isn't reducible to game payoffs. The same game can sustain many norms. That's why Henrich's transmission story is necessary."

**Henrich**, smiling: "Cristina just admitted my point."

---

## §2 — Where does $\lambda$ live? (the path narrows through cedars)

The trail enters denser timber. The light dims. **Hyland** has been quiet, now speaks.

> "I want to be careful here. The $\lambda$ in our 2025 paper tempers the *variational cost of revising a prior*. It's a free-energy quantity. To put it into Jonas's model, I need to know *which prior* it tempers.
>
> "Three candidates, each different.
>
> "**Candidate 1**: $\lambda$ tempers the prior over actions in the stage game. Agent has $C \in \mathbb{R}^{|\mathcal{A}|}$, log-preferences over actions. Updating $C$ is costly — high $\lambda$ means agent persists in the same action even as evidence (reputation feedback) accumulates against it. This is closest to the v02 implementation Jonas has.
>
> "**Candidate 2**: $\lambda$ tempers the prior over *what others will do* — the descriptive belief in Bicchieri's sense. Agent has a model of the population's action distribution; updating that model is costly. Conservatism here means slow updating of social statistics, even after observing many neighbours.
>
> "**Candidate 3**: $\lambda$ tempers the prior over the *generative-model parameters* — the agent's belief about the structure of reputation, which actions deserve approval. This is the deepest level. Conservatism here means persisting in a frame about *what counts as appropriate*, not just doing what's locally rewarded.
>
> "These are not the same object and they predict different cascade dynamics."

**Ramstead**, who has been walking last, finally speaks:

> "David, I'd add a fourth. $\lambda$ tempers the prior at the *Markov blanket level* — the community's shared generative model. Individual agents in a community have correlated priors not because they were copied independently, but because they share a partition. The cost of mind-change is asymmetric across the blanket: revisions that align with the community are cheap; revisions that defect from the community are expensive. This is closer to Henrich's cumulative-cultural account than to a single-agent variational cost.
>
> "When you ask whether $\lambda$ is per-agent or per-community, the right answer is *both*: there's an individual-scale free-energy gradient and a community-scale one, and they interact through the blanket statistics."

**Hyland**: "Max, I appreciate the integration but I'd want to formalise what *Markov-blanket-level conservatism* actually computes. Otherwise it's a metaphor."

**Ramstead** laughs: "Fair. The closest formal object I can give you is: the community's average prior $\bar{C} = \frac{1}{|S|}\sum_{i \in S} C_i$, and individual cost of mind-change includes a term $\lambda_{\text{social}} \cdot \mathrm{KL}[C_i \| \bar{C}]$. So agents pay a free-energy cost when their prior deviates from their community's mean. That's a precise version of conformity bias as variational cost."

**Henrich**: "Now you're speaking my language. The conformity bias I document empirically — Boyd and Richerson's payoff-biased and frequency-biased transmission — comes out of that formulation. With the right cost structure, you get within-group homogeneity and between-group variance, which is *exactly* what cumulative cultural evolution requires."

**Leibo**: "But you need to instantiate this in a *game*. A KL term to a community mean is just a coordination penalty. What does the community mean track? The community mean of *what action profile*? In what *game*? Without those, you're moving costs around without selection pressure."

---

## §3 — The clearing (synthesis attempt)

The trail opens into a meadow with a granite outcrop. We sit on the rock; the sun is fully out now. Below us, mist still hangs in the valley.

**Me**: "Let me try to integrate. We have:

- **Bicchieri**: norms are conditional preferences sustained by mutual expectations. The relevant Bayesian object is the agent's *belief about the population's action distribution* and *belief about the population's normative expectations*.
- **Henrich**: norms are transmitted by prestige and conformity. The relevant social-learning operator is *not equal-weight pooling over neighbours* but *prestige-weighted asymmetric copying*.
- **Leibo**: norms select equilibria in repeated games. The model needs a stage game with multiple equilibria and a reputation system that aggregates observed actions.
- **Hyland**: $\lambda$ tempers a specific prior. Candidates 1–4 give qualitatively different dynamics.
- **Ramstead**: norms are shared generative models; conformity is variational cost on deviation from community priors.

"Can these compose into one model?"

**Bicchieri** picks at the moss on the rock:

> "Maybe. Here's a sketch. Each agent has *three* posteriors:
>
> "(i) **What others do** ($\hat{p}_{\text{descriptive}}$): a belief over the population's action distribution. Updated from observed neighbour actions.
>
> "(ii) **What others think one should do** ($\hat{p}_{\text{injunctive}}$): a belief over the population's normative expectations. Updated from observed sanctions — when did neighbours give or withhold reputation?
>
> "(iii) **One's own preference under the norm** ($p_{\text{self}}$): a conditional preference. Has the form: *I prefer action $a$ if both descriptive and injunctive expectations support $a$, otherwise I prefer some default*.
>
> "Action selection samples from $p_{\text{self}}$. Reputation payoff comes from neighbours' (ii) judging the agent's action."

**Henrich** nods slowly:

> "Yes, but the *update* on (i) and (ii) shouldn't be Bayes-flat. Some neighbours are more informative than others — those who pay closer attention to community standards, those whom others themselves copy. Add prestige weights $\pi_{ij}$: agent $i$'s update on $\hat{p}_{\text{descriptive}}$ from observing $j$'s action is weighted by $\pi_{ij}$, where prestige is observable from *patterns of who copies whom*. Higher-$\pi$ individuals contribute more to belief revision. This generates within-group homogeneity that random-mixing can't."

**Leibo**:

> "OK, this is starting to be operational. But I want to add: the stage game can't be one-shot. Make it repeated, with reputation accumulating. Specifically:
>
> "Each round, each agent encounters the situation, picks an action $a_i \in \mathcal{A}$. Neighbours observe $a_i$, compute reputation update $\rho_{ij,t+1} = \rho_{ij,t} + r(a_i; \hat{p}_{\text{injunctive}, j})$, where $r$ is positive when $a_i$ matches what $j$ thinks is appropriate, negative otherwise. Resources flow proportional to total reputation $\sum_j \rho_{ij}$.
>
> "Now you have the tragedy-of-commons-flavoured tension: actions that are individually rewarding (e.g., scold — low cost, no engagement) might be reputationally punished if community injunctive norms hold them inappropriate. The game has multiple equilibria depending on which injunctive norm is locked in. Cascade dynamics happen when injunctive expectations shift; coordination on the new equilibrium then proceeds."

**Hyland**:

> "And here's where $\lambda$ enters. I want $\lambda$ to act on (iii) — the agent's own preference. Updating $p_{\text{self}}$ is costly proportional to $\lambda_i$. Updating (i) and (ii) is *not* costly — those are just Bayesian descriptive beliefs about the social world.
>
> "This gives you a clean separation. Heterogeneity in $\lambda_i$ controls *who is willing to deviate* from the local norm even when descriptive evidence suggests the norm is shifting. Low-$\lambda$ agents adjust $p_{\text{self}}$ quickly to match $\hat{p}_{\text{descriptive}}$ — they're conformist, but quickly. High-$\lambda$ agents stick to their old preferences regardless. Low-$\lambda$ agents are the *cascade seeds*; high-$\lambda$ agents are the *cascade resistors*.
>
> "The headline empirical phenomenon: in the cascade dynamics, low-$\lambda$ agents in high-prestige network positions seed transitions to new equilibria; high-$\lambda$ agents in low-prestige positions are the lock-in residue."

**Ramstead**:

> "And the multicultural-equilibria result Jonas wants comes for free: different network communities have different lock-in trajectories because their internal $\lambda$ distributions × prestige networks × initial norm states put them in different basins of attraction. The shared generative model is the equilibrium they coordinated on; cost-of-mind-change keeps them there even when payoff drifts.
>
> "What I'd add: the fact that communities have *different* equilibria isn't just initial-condition variance. It can be selected. Communities with higher mean $\lambda$ have stickier coordination — more robust to noise, slower to track drift. Under environmental change, low-$\lambda$ communities outcompete on tracking accuracy in the short term but lose to high-$\lambda$ communities on coordination stability in the long term. That's a non-obvious prediction worth testing."

---

## §4 — Tensions left unresolved (walking back)

The sun is past noon now. We start back down the trail.

**Bicchieri**: "I'll grant the integration is coherent. But I'm uneasy about $\lambda$ on the agent's *own* preferences (iii). In my account, agents don't have non-norm-conditioned preferences in the relevant situations. Take the norm away, the conditional preference vanishes — there's nothing for $\lambda$ to temper. What is the agent *resisting* changing, exactly?"

**Hyland**: "A reasonable question. In our model, $p_{\text{self}}$ is a prior over policies, conditioned on the agent's *current model* of $\hat{p}_{\text{descriptive}}$ and $\hat{p}_{\text{injunctive}}$. Even when those expectations update, the conditional itself — the *function* mapping expectations to action preferences — has inertia. That's what $\lambda$ tempers. It's not preference under no norm; it's the conditional structure linking norm-state to action."

**Bicchieri**: "Acceptable. But it's a strong cognitive commitment. You're saying agents have explicit conditional structures stored. I'd want behavioural predictions that distinguish your model from a simpler one where agents just sample from $\hat{p}_{\text{descriptive}}$ with some conformity bias."

**Henrich**: "I have a different worry. Your prestige weights $\pi_{ij}$ — where do they come from? In my work, prestige is *paid* by observers freely; you can detect it by counting copying patterns. In your model, you'd need agents to maintain a posterior over each neighbour's prestige, which is another Bayesian object. That's expensive."

**Leibo**: "Practically, prestige can be approximated as eigenvector-centrality on the copying graph. Each agent doesn't need a full posterior over each neighbour's prestige; they need a low-dimensional summary. We do this in MARL all the time."

**Ramstead**: "The question that doesn't have a clean answer yet: at what scale do reputations live? If reputation is per-pair $\rho_{ij}$, you have $N^2$ state variables — too many. If reputation is per-agent $\rho_i$ (shared across all neighbours), you lose the local structure. The right answer is probably per-community: $\rho_{i, c}$ where $c$ is community membership. But that imports community structure as a primitive, which is what the model is trying to *explain*. Bootstrapping issue."

**Hyland**: "Two-pass model. Initialise communities from network structure (e.g., spectral clustering). Run dynamics. Update community membership periodically based on revealed coordination patterns. Iterate."

**Leibo**: "Or: don't reify communities. Just have per-pair reputation that's computationally tractable for moderate $N$. With $N = 30$, $N^2 = 900$ — fine."

The trail is descending now. The valley opens up below us.

---

## §5 — The synthesis (back at the trailhead)

Standing at the trailhead, looking over a pondside marsh:

### What everyone agrees on

1. The model needs a **stage game** with multiple discrete actions in a specific micro-situation. The beggar example is fine; tragedy-of-commons is fine; what matters is *multiple equilibria*.
2. Reputation must be **action-channel observed**, not posterior-channel observed. Agents see what neighbours *do*, not what they *believe*.
3. The agent maintains **at least two Bayesian objects**: a belief about *what others do* (descriptive) and a belief about *what others think one should do* (injunctive). These update via Bayes from observation.
4. **Prestige weights** modulate social-learning updates. Equal-weighted Bayes pooling is too thin.
5. **$\lambda$ acts on the conditional preference** — the function mapping (descriptive, injunctive) expectations to action preferences. This is what's costly to revise.
6. The headline phenomenon is **multicultural equilibria with hysteresis**. Different communities lock onto different action equilibria; high-$\lambda$ communities are slower to escape their basin even under payoff drift.

### Where they disagree (still open)

- **Bicchieri vs. Hyland** on whether agents have non-conditional preferences at all. Resolved instrumentally by treating $p_{\text{self}}$ as a *conditional structure* whose inertia $\lambda$ tempers.
- **Henrich vs. Leibo** on whether equilibrium analysis or transmission dynamics is the central object. Resolved by including both: the game gives the basins; transmission selects between them.
- **Ramstead vs. Leibo** on whether community is a primitive or a derived quantity. Unresolved; punt to the bootstrapping approach (initialise from graph structure, refine via revealed coordination).
- **What is reputation, formally?** Per-pair, per-community, per-agent — different structural choices. For $N = 30$, per-pair is computationally fine; for larger $N$, per-community becomes attractive.

### What this means for the model spec

A revised model object:

```
For each agent i:
  C_i in R^|A|                    # log-preferences over actions (the prior λ tempers)
  hat_p_desc_i in Δ(A)^N          # belief over each neighbour's action distribution
  hat_p_inj_i  in Δ(A)^N          # belief over each neighbour's normative expectations
  pi_i in R_+^N                   # prestige weights toward each neighbour
  rho_ij in R                     # pairwise reputation balance

Stage game:
  Each round t:
    each agent picks a_i ~ softmax(C_i)
    neighbours observe a_i
    each neighbour j updates ρ_ij based on r(a_i; hat_p_inj_j)
    each agent observes (a_j) for j in N(i), updates hat_p_desc_i and hat_p_inj_i
    C_i updates via λ-tempered Eq.13 toward an evidence target derived from
      hat_p_desc and hat_p_inj
    π_ij updates based on copying-success patterns
    resources flow proportional to total reputation
```

This is a substantial reformulation of the brief. The $(\lambda, \text{drift rate})$ phase diagram becomes $(\lambda, \text{prestige asymmetry}, \text{environmental drift in payoffs})$ — a 3D phase space with richer phenomenology. The dissociation from rule-induction (Oldenburg & Zhi-Xuan 2024) becomes sharper because their model has no notion of conditional preferences.

### Headline empirical claims (to test)

1. **Multicultural equilibria**: under heterogeneous initial conditions and high mean $\lambda$, distinct network communities lock onto distinct action equilibria; the configuration is stable under moderate environmental drift.
2. **Cascade asymmetry**: low-$\lambda$ agents in high-prestige positions seed transitions; the same agents in low-prestige positions don't.
3. **Stability-tracking trade-off**: low-$\lambda$ communities track payoff drift more accurately on short timescales but coordinate less robustly under noise; high-$\lambda$ communities show the reverse. There's a sweet spot.
4. **Reputation as observable**: the dissent front in the network is visible *first* in $\rho_{ij}$ time series before it shows up in action distributions. This is the early-warning signal.

### What's genuinely novel

Relative to existing work:

- **Henrich-style cumulative cultural evolution** typically uses non-strategic learning models (vertical and horizontal transmission with biases). Adding the variational cost of mind-change — an internal cognitive constraint — gives a microfoundation for *why* certain norms entrench faster than others.
- **Bicchieri-style norm theory** treats expectations as observable equilibrium objects but doesn't have a dynamics for *which* equilibrium a community lands on. Adding $\lambda$-conservatism + prestige asymmetry generates dynamics.
- **Leibo-style MARL** has equilibrium selection in repeated games but typically through reinforcement learning. Active inference gives a richer cognitive substrate that connects to single-agent variational results.
- **Active-inference culture (Ramstead et al.)** has shared generative models but lacks the asymmetric-cost mechanism that produces hysteresis. $\lambda$-conservatism plugs that gap.

The novelty is in the *integration*, not in any one component. This is also the risk: integration papers can read as derivative if the integration doesn't yield a non-obvious empirical prediction. Predictions (1)–(4) above are candidates for the headline result.

---

## §6 — What needs verification

This walk produced a Stage 1–2 output. To advance the design, the user should:

- **Read** Bicchieri 2017 (*Norms in the Wild*) ch. 1–2 for the conditional-preference formalism.
- **Read** Henrich 2015 (*The Secret of Our Success*) ch. 2–4 for transmission dynamics with biases.
- **Read** Albarracin et al. 2022 (*Epistemic Communities Under Active Inference*) for the AIF community formalism — the precursor that this work most directly extends.
- **Look at** Joel Leibo's recent work on norm emergence (e.g., Vinitsky et al. 2023, "A learning agent that acquires social norms from public sanctions in decentralized multi-agent settings").
- **Send to David Hyland**: the choice of $\lambda$ acting on conditional preferences (iii), not on descriptive beliefs (i)/(ii). Confirm this matches his variational intent.
- **Test computational tractability** with $N = 30$, $|\mathcal{A}| = 5$, full pairwise reputation: this is well within budget for the IWAI deadline.

## §7 — Decision points the user faces

1. **Stay with the v02 architecture or pivot to this redesign?** The v02 has the architecture-fix path documented in `architecture_review.pdf`. The redesign is a more ambitious model. The redesign is *more interesting* but takes the model further from the brief David already saw.
2. **If pivoting, when?** A pivot now means the May 14 (E1) deadline is at risk; a pivot after E1 means E1 results are with the simpler model and the redesign is for E2–E4.
3. **Talk to Hyland before pivoting?** He's the natural co-author; the redesign extends his formalism in a direction he hasn't published. Sending him this synthesis before committing is probably wise.

The walk doesn't decide these for you. It just makes the trade-offs visible.

— end of walk
