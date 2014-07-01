'use strict';

var UserFactory = angular.module('userFactory', []);

UserFactory.factory('User', function (Alerts) {
    var _user = {name: "Barry Allen", coins: 333, debt: 0, message: ""};

    return {
        toDefault: function () {
            _user = angular.copy({name: "Barry Allen", coins: 333, debt: 0, message: ""});
        },
        user: function () {
            return _user;
        },
        addDebt: function (amount) {
            _user.coins += amount;
            _user.debt += amount;
        },
        exitMarket: function () {
            var balance = _user.coins - _user.debt;
            return ((balance > 0) && balance > _user.debt) ?
                   "You made: \u0e3f" + balance: "You owe: \u0e3f" + balance;
        }
    };
});
