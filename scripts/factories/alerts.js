'use strict';

var AlertsFactory = angular.module('alertsFactory', []);

AlertsFactory.factory('Alerts', function () {
    var _alerts = [];

    return {
        addAlert: function (msg, type) {
            if (_alerts.indexOf(msg) === -1) {
                _alerts.push({msg: msg, type: typeof type === "undefined" ? "info" : type});
            }
        },
        closeAlert: function (index) {
            _alerts.splice(index, 1);
        },
        alerts: function () {
            return _alerts;
        }};
});
