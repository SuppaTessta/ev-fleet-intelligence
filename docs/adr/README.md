# Architecture Decision Records

Decisions worth the reader's time — mostly the ones where the answer changed, or where an
attractive option was rejected for a reason that is not obvious from the code.

| # | Decision | Why it is here |
|---|---|---|
| [0001](0001-maintenance-lp-vs-greedy.md) | Keep the maintenance LP, publish that it does not beat greedy | The 37.5% improvement was a bug in its own baseline |
| [0002](0002-rejected-eviot-dataset.md) | Reject the EVIoT telemetry dataset | It has exactly the schema this project wanted, and is random noise |
| [0003](0003-casting-split-leakage.md) | Re-split the casting dataset before augmentation | 97.5% of test parts were in training, hiding a 28% false-reject rate |
| [0004](0004-streamlit-over-react.md) | Streamlit, refactored, over a React rewrite | Scope discipline on a solo project |

An ADR here is expected to state what was measured, not just what was decided.
