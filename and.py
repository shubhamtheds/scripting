import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot

def main(data, modelName, plotName, eta, epochs):

    df_AND = pd.DataFrame(AND)

    X, y = prepare_data(df_AND)

    
    perceptron_AND = Perceptron(eta = ETA, epochs = EPOCHS)
    perceptron_AND.fit(X, y)

    _ = perceptron_AND.total_loss()

    perceptron_AND.save(filename="AND.model")  
    reload_AND = Perceptron().load(filepath = "model/AND.model")

    save_plot(df_AND, reload_AND, filename="AND.png")


if __name__ == "__main__":

    AND = {
        "x1" : [0, 0, 1, 1],
        "x2" : [0, 1, 0, 1],
        "y" : [0, 0, 0, 1]
    }

    ETA = 0.3,
    EPOCHS = 10

    main(data=AND, modelName="AND.model", plotName="AND.png", eta=ETA, epochs=EPOCHS)