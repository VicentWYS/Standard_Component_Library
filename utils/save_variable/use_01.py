from utils.get_data import generate_random_seed
from utils.save_load import save_variable, load_variable

seed1 = generate_random_seed(100)

save_variable(seed1, './save/seed/seed1')

seed1 = load_variable("./save/seed/seed1")
