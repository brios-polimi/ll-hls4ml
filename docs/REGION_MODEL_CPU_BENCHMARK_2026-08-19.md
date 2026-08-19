# Region model CPU benchmark

One real `3layer` validation graph, one graph per batch, 1 Torch CPU thread(s), median of 5 timed forwards after 2 warmups.

Tensor sizes: 3,551 instructions, 253 blocks, 15 loops.

| model | parameters | median ms |
| --- | --- | --- |
| H0 | 239,568 | 13.6 |
| region / mean messages / generic | 301,968 | 15.6 |
| region / cardinality / generic | 634,896 | 75.0 |
| region / mean messages / hardware | 323,476 | 22.1 |
| region / cardinality / hardware | 656,404 | 79.2 |

This is a feasibility/accounting benchmark, not an accuracy result.
