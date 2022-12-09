# Input trace data

For information on the provided time series, see `azure_demand/LICENSE`.

For the schema of the time series files, see `azure_demand/schema.txt`.

### How examples are created

Two types of files are used: raw time series files, and "offset"
files.  Lines in the offset file provide an index into the raw time
series.  One "window" (consisting of a conditioning range and
prediction range) is extracted at each offset.  Windowed examples are
created in order of the offsets in the offset file.  Different offsets
are used for training/validation/testing, as indicated by the file
suffixes.  In addition, when training/testing, we pass arguments named
`--train_loss_end_pt_s`, `--test_loss_start_pt_s`, and
`--test_loss_end_pt_s` to further define which elements of the window
are included in the loss, ensuring no information from the validation
and testing periods are available when training, and no information
from the testing period is available when validating.

Examples of using the input data are provided in train_test.sh
