from QBO_GIF_Generator import BayesianGIFGenerator
from QBO_Image_Generator import BayesianImageGenerator

file_format = 'gif'

if file_format == 'image':
    model = BayesianImageGenerator(
        file_path="MerlijnFriso3_250x250_Verbeterd.csv",
        model="Quantum",
        mode="Value",
        n_start=1,
        n_end=100
    )
    model.run_optimizer()
    model.plot_iteration(2)

elif file_format == 'gif':
    BayesianGIFGenerator(
        file_path='MerlijnFriso3_250x250_Verbeterd.csv',
        model='Quantum',
        mode='Value',
        n_start=10,
        n_end=100,
        n_frames=20,
        gif_filename='Quantum_progress.gif',
        gif_duration=10
    )
else:
    raise ValueError(f"File format {file_format} doesn't exist! Available formats: image or gif")