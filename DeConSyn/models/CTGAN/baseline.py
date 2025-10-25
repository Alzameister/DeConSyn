from DeConSyn.data.data_loader import DatasetLoader
from DeConSyn.models.CTGAN.synthesizers.ctgan import CTGAN
from DeConSyn.training_framework.fsm.fsm_behaviour import discrete_cols_of
from DeConSyn.training_framework.start import ADULT_PATH, ADULT_MANIFEST
from DeConSyn.io.io import get_repo_root

# Train adult dataset on default CTGAN settings
adult = ADULT_PATH
manifest = ADULT_MANIFEST
loader = DatasetLoader(f"{adult}/{manifest}")
full_train = loader.get_train()
full_test = loader.get_test()
discrete_columns = discrete_cols_of(full_train)

model = CTGAN(verbose=True)
model.fit(full_data=full_train, train_data=full_train, discrete_columns=discrete_columns)
# Save model
root = get_repo_root()
path = root / "runs" / "baseline_ctgan" / "ctgan_adult_default.pkl"
path.parent.mkdir(parents=True, exist_ok=True)
model.save(path)