'use strict';

var EmbarkCtrl = angular.module('embarkCtrl', []);

EmbarkCtrl.controller('EmbarkCtrl', function ($scope, Alerts) {
    $scope.runAlready = false;
    if (!$scope.runAlready) {
        Alerts.addAlert(''.concat("Welcome! - Goal: compete against other users of your browser in the race to: ",
                                  "develop and then exit a business ethically and profitably."));
        $scope.runAlready = true;
    }
});
