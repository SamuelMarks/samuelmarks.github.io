'use strict';

var AlertsFactory = angular.module('alertsFactory', []);

AlertsFactory.factory('Alerts', function () {
    var _alerts = [];

    return {
        addAlert: function (msg, type) {
            var alert = {msg: msg, type: typeof type === "undefined" ? "info" : type};
            var exists = false;
            _alerts.forEach(function (elem) {
                if (elem.msg === alert.msg) exists = true;
            });
            // ^ indexOf wasn't working, so this workaround ^
            if (!exists) _alerts.push(alert);
        },
        closeAlert: function (index) {
            _alerts.splice(index, 1);
        },
        alerts: function () {
            return _alerts;
        }};
});
