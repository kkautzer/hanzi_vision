<!-- # TODO - Commit Tests -->

## Tests for `model/data/**`
- [x] Compilation/Clean Exit: `dataset.py`

## Tests for `model/train.py` and `model/model.py`
- [x] Compilation/Clean Exit: `model.py`
- [ ] Compilation/Clean Exit: `train.py`
- [ ] Create model w/ default parameters: `train.py`
- [ ] Create model w/ non-default parameters: `train.py`
- [ ] Resume training from best: `train.py`
- [ ] Resume training from specified epoch: `train.py`

## Tests for `model/eval/**`
- [ ] Compilation/Clean Exit: `eval_custom_batch.py`
- [ ] Compilation/Claen Exit: `eval_custom_single.py`
- [ ] Compilation/Clean Exit: `test.py`

## Tests for `model/scripts/**`
- [ ] Compilation/Clean Exist: `convert_to_binary_image.py`
- [ ] Compilation/Clean Exit: `create_csv_test.py`
- [ ] Compilation/Clean Exit: `create_filtered_set.py`
- [ ] Compilation/Clean Exit: `crop_image.py`
- [ ] Compilation/Clean Exit: `generate_whitelist.txt`

## Tests for `server/**`
- [ ] Compilation/Run: `app.py`
- [ ] Correct Response: `app/` endpoint
- [ ] Valid Response: `app/evaluate` endpoint - valid input, first model
- [ ] Valid Response: `app/evaluate` endpoint - valid input, second model
- [ ] Clean Response: `app/evaluate` endpoint - invalid input file
- [ ] Clean Response: `app/evaluate` endpoint - missing input file
- [ ] Clean Response: `app/evaluate` endpoint - invalid request parameters
- [ ] Correct Response: `app/models` endpoint - valid input
- [ ] Clean Response: `app/models` endpoint - invalid request parameters
- [ ] Correct Response: `app/models/data/<model_name>` endpoint - valid input (first model)
- [ ] Correct Response: `app/models/data/<model_name>` endpoint - valid input (second model)
- [ ] Clean Response: `app/models/data/<model_name>` endpoint - invalid request parameters
- [ ] Correct Response: `app/characters` endpoint - valid input
- [ ] Clean Response: `app/characters` endpoint - invalid request parameters
- [ ] Correct Response: `app/characters/<character>` endpoint - valid input (first char)
- [ ] Correct Response: `app/characters/<character>` endpoint - valid input (second char)
- [ ] Clean Response: `app/characters/<character>` endpoint - invalid input parameters
