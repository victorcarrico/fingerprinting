import csv

import numpy as np
from sklearn.metrics import mean_squared_error

from constants import ERBS
from helpers import distance
from PyRadioLoc.Pathloss.Models import (
    FreeSpaceModel, FlatEarthModel, LeeModel, EricssonModel,
    Cost231Model, Cost231HataModel)

PATHLOSS_MODELS = [
    FreeSpaceModel, FlatEarthModel, LeeModel, EricssonModel,
    Cost231Model, Cost231HataModel]

PATHLOSS_MODELS = [Cost231Model]


class BestPathlossCSV:

    def __init__(self, band, eirp, file, output):
        self.band = band
        self.eirp = eirp
        self.file = file
        self.output = output

    def calc_errors(self):
        with open(self.file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            distances = []
            errors_point = []
            for row in csvreader:
                point_localization = (float(row[0]), float(row[1]))
                rssis = [float(rssi) for rssi in row[2:]]
                distances = [
                    distance(erb, point_localization)
                    for erb in ERBS]
                errors_models_point = {}
                for pathloss_model in PATHLOSS_MODELS:
                    m = pathloss_model(1800)
                    pathloss_predictions_erbs = m.pathloss(distances)
                    pathloss_real_erbs = [self.eirp - rssi for rssi in rssis]
                    mean_squared_error_erbs = np.sqrt(
                        mean_squared_error(pathloss_real_erbs, pathloss_predictions_erbs))
                    errors_models_point[pathloss_model.__name__] = mean_squared_error_erbs
                errors_point.append(errors_models_point)
            return errors_point

    def find_best_error(self):
        sum_errors = {}
        errors = self.calc_errors()
        for point in errors:
            for model_name, error in point.items():
                partial_sum = sum_errors.get(model_name, 0)
                sum_errors[model_name] = partial_sum + error
        return sum_errors
