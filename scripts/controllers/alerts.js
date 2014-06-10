'use strict';

var AlertsCtrl = angular.module('alertsCtrl', []);

AlertsCtrl.controller('AlertsCtrl', function ($scope, Alerts) {
    $scope.alerts = Alerts.alerts();
    $scope.closeAlert = Alerts.closeAlert;
});
