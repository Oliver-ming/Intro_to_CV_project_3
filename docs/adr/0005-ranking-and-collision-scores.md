# Ranking uses pairwise order and AbsRel; collision is a frame-level warning

Ordinal depth papers score pairs, not Spearman. Object-distance papers lead with AbsRel. This project reports pairwise order accuracy and obstacle AbsRel. A collision warning is correct or not per frame, scored with precision, recall, and F1, because the rule is a fixed distance and the clips have no velocity for time-to-collision. The nearest obstacle's AbsRel is reported beside that.

Mean absolute error in meters and Spearman were the alternatives. They are not the headline numbers in the papers this protocol follows.
