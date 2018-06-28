import csv

from utils import haversine
from PyRadioLoc.Pathloss.Models import (
    FreeSpaceModel, FlatEarthModel, LeeModel, EricssonModel,
    Cost231Model, Cost231HataModel, OkumuraHataModel, Ecc33Model,
    SuiModel)


ERB1 = (-8.068361111, -34.892722222)
ERB2 = (-8.075916667, -34.894611111)
ERB3 = (-8.076361111, -34.908)
ERB4 = (-8.075916667, -34.8946111116)
ERB5 = (-8.066, -34.8894444444444)
ERB6 = (-8.06458333333333, -34.8945833333333)


class PathlossCSV:
    def __init__(self, band, eirp, file, output):
        self.band = band
        self.eirp = eirp
        self.file = file
        self.output = output

    def calc(self):
        with open(self.file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            distances = []
            for row in csvreader:
                point_localization = (float(row[0]), float(row[1]))
                distance = haversine.distance(ERB1, point_localization)
                distances.append(distance)
        return distances
