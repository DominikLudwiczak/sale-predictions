import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import math
from IPython.display import display

class Viz:
    def __init__(self) -> None:
        self.forecasts = pd.read_csv("../datasets/forecast.csv")
        self.forecasts[(self.forecasts.productdictionaryid == 1)]
        self.forecasts = self.forecasts[['pointofsaleid', 'forecastvalue', 'forecastdate']]
        self.forecasts.sort_values(by=['forecastdate', 'pointofsaleid'], inplace=True)
        self.forecasts = self.forecasts.groupby(['forecastdate', 'pointofsaleid'], as_index=False).median()

    def show_sales(self, predictions, true_values, points_of_sales, show_product):
        if len(points_of_sales) == 1:
            plt.figure(figsize=(25, 20))
        else:
            plt.figure(figsize=(50, 50))
        for p in range(len(predictions)):
            for point in range(len(points_of_sales)):
                true_val = true_values[p][true_values[p].id == points_of_sales[point]].reset_index(drop=True)
                pred_val = predictions[p][predictions[p].id == points_of_sales[point]].reset_index(drop=True)
                if len(points_of_sales) == 1:
                    plt.subplot(math.ceil(len(predictions)/4), 4, p+1)
                else:
                    plt.subplot(len(predictions), len(points_of_sales), (p * len(points_of_sales)) + point+1)
                plt.title('point: '+str(points_of_sales[point])+' k= '+str(p+1))
                plt.plot(true_val.index, true_val[show_product], color='blue', label='Actual Sale')
                plt.plot(pred_val.index, pred_val[show_product], color='red', label='Predicted Sale')
                plt.xlabel('Date')
                plt.ylabel('Sale')
                idx = np.arange(-1, len(true_val.index), 10)
                idx[0] = 0
                plt.xticks(idx, true_val[true_val.index.isin(idx)].dzien_rozliczenia, rotation=45)
                plt.legend()
        plt.show()

    def show_losses(self, losses, column):
        plt.figure(figsize=(20, 10))
        for l in range(len(losses)):
            plt.subplot(1, 4, l+1)
            plt.title(losses[l][0])
            plt.plot(losses[l][1].index+1, losses[l][1][column], color='blue', label='Loss')
            plt.xlabel('Fold')
            plt.ylabel('Loss')
            plt.legend()
        plt.show()

    def show_losses_for_KNN(self, losses):
        plt.figure(figsize=(25, 25))
        for l in range(len(losses)):
            plt.subplot(4, 4, l+1)
            plt.title('(Rolling CV) k= '+str(l+1))
            best_K = min(range(len(losses[l])), key=list(losses[l]['RSME (Chicken)']).__getitem__)
            plt.plot(losses[l].index, losses[l].values, color='blue', label='RSME (Chicken)')
            plt.xlabel('K')
            plt.ylabel('Loss')
            plt.plot(best_K+1, losses[l].iloc[best_K], 'ro', label='Best K - nearest neighbour')
            idx = np.arange(-1, len(losses[l].index), 10)
            idx[0] = 0
            plt.xticks(idx, idx+1)
            plt.legend()
        plt.show()

    def show_error_summary(self, errors, column, method_name, titles):
        num_errors = len(errors)
        rows, cols = math.ceil(len(errors)/4), 4
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharey=True)

        base_cmap = plt.cm.get_cmap('tab20')
        colors = base_cmap.colors[:num_errors]

        for i, (ax, error, title) in enumerate(zip(axes.flatten(), errors, titles)):
            ax.plot(error.index + 1, error[column], color=colors[i])
            ax.set_xlabel('Fold', fontsize=15)
            if i % cols == 0:
                ax.set_ylabel(f'Loss {method_name}', fontsize=15)
            ax.set_title(f'Error - {title}', fontsize=15)
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()

    def show_models_comp(self, model_errors, own_errors, method_name):
        plt.title(f'{method_name} - Models comparison')
        plt.plot(model_errors.index + 1, model_errors.product1, color='blue', label='Model Loss')
        plt.plot(own_errors.index + 1, own_errors.product1, color='red', label='Our Model Loss')
        plt.xlabel('Fold')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def show_estimations(self, k, predictions, test_sets, product, isMobile=False):
        plt.figure(figsize=(25, 20))
        for i in range(k):
            if isMobile:
                merged_df = test_sets[i]
            else:
                forecasts = self.forecasts[(self.forecasts.forecastdate.isin(test_sets[i].dzien_rozliczenia) & (self.forecasts.pointofsaleid.isin(test_sets[i].id)))]
                merged_df = pd.merge(test_sets[i], forecasts, left_on=['dzien_rozliczenia', 'id'], right_on=['forecastdate', 'pointofsaleid'], how='inner')
                merged_df = merged_df[['id', 'dzien_rozliczenia', 'product1', 'forecastvalue', 'product2']]
            merged_df = pd.merge(merged_df, predictions[i], left_on=['dzien_rozliczenia', 'id'], right_on=['dzien_rozliczenia', 'id'], how='inner')
            merged_df.rename(columns={'product1_x': 'product1', 'product1_y': 'prod1_predicted'}, inplace=True)
            merged_df.rename(columns={'product2_x': 'product2', 'product2_y': 'prod2_predicted'}, inplace=True)
            merged_df.fillna(0, inplace=True)
            plt.subplot(math.ceil(k/4), 4, i+1)
            plt.title('k= '+str(i+1))
            if product == 'product1' and not isMobile:
                plt.scatter(merged_df.product1, merged_df.prod1_predicted, color='red', marker='o', label='Predicted')
                plt.scatter(merged_df.product1, merged_df.forecastvalue, color='green', marker='o', label='Current model')
            else:
                plt.scatter(merged_df.product2, merged_df.prod2_predicted, color='red', marker='o', label='Predicted')
            
            plt.plot(merged_df[product], merged_df[product], color='blue', linestyle='-', linewidth=2, label='Actual')
            plt.xlabel('Actual Sale')
            plt.ylabel('Predicted Sale')
            plt.legend()
