# Phase 3 — Shape optimization + dashboard
*Planned · 2027*

**Goal.** Use the fast surrogate from Phase 1 inside an **optimization loop** to search airfoil
shapes / operating points for the best lift-to-drag, then expose the results in an interactive
**dashboard**.

**Why it needs the surrogate.** Optimization evaluates thousands of candidate designs — infeasible
with hours of CFD each. A millisecond surrogate makes the search practical. This phase is the
pay-off of the whole roadmap.
