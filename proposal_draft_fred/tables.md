|           | labels               |                filter |                   use |
|-----------|:---------------------|----------------------:|----------------------:|
| DAWN      | car, bus, truck      | keep: fog, snow, rain | validation + training |
| EU-DETRAC | car, bus, van, other |           keep: sunny |              training |



| measurement      | training     | validation |
|------------------|--------------|------------|
| baseline         | none         | DAWN-test  |
| augmented images | DETRAC-train | DAWN-test  |


| measurement      | training     | validation |
|------------------|--------------|------------|
| baseline         | none         | DAWN-test  |
| actual AWC images | DAWN-train | DAWN-test  |

