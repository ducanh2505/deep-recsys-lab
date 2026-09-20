# Bind datasets to problems explicitly

A Dataset preserves canonical source observations independently of Recommendation Problems,
and an Experiment connects it to exactly one Problem through an immutable Problem–Dataset
Binding. The Binding maps entity and field roles, preserves their declared meanings, and
validates compatibility without transforming content or defining Preference Signals or
Evaluation Judgments. Giving each Problem its own Dataset identity would simplify local
schemas but duplicate the same evidence and mix task interpretation into data identity;
requiring one universal Lab-wide interaction schema would instead hard-code current
user-item assumptions into future Problems. The explicit Binding adds a versioned object and
validation step in exchange for reusable Datasets and Problem-local contracts. Any mapping
that must alter content or discard source semantics is Dataset Adaptation and produces a new
Dataset rather than being hidden in the Binding.
