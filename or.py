import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot

def main(data, modelName, plotName, eta, epochs):

    df_OR = pd.DataFrame(OR)

    X, y = prepare_data(df_OR)

    
    perceptron_OR = Perceptron(eta = ETA, epochs = EPOCHS)
    perceptron_OR.fit(X, y)

    _ = perceptron_OR.total_loss()

    perceptron_OR.save(filename="or.model")  
    reload_OR = Perceptron().load(filepath = "model/or.model")

    save_plot(df_OR, reload_OR, filename="or.png")


if __name__ == "__main__":

    OR = {
        "x1" : [0, 0, 1, 1],
        "x2" : [0, 1, 0, 1],
        "y" : [0, 1, 1, 1]
    }

    ETA = 0.3,
    EPOCHS = 10

    main(data=OR, modelName="or.model", plotName="or.png", eta=ETA, epochs=EPOCHS)