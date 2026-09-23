# Evaluate Empty-History with new Subjects

The Empty-History quality cohort uses Subjects whose first Positive Interaction appears in the
Future Window and who have no Positive Interactions in the Data Snapshot. We chose this over
masking existing Subjects' histories because the serving Query truly has no usable preference
history, and masking would change the question being measured. Popularity is the only route;
when no such Subjects exist, quality is n/a with denominator zero.
