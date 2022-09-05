#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*1000000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*950000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*800000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*750000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*900000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*850000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*700000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*650000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*600000*
#rm ./model_data/dcase2020_fuss/baseline_dry_train_backup/**/*550000*

python3 makeinputparams.py
python3 makeinputs_disc.py
python3 makeconfig.py
python3 makeconfig_disc.py