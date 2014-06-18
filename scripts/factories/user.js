'use strict';

var UserFactory = angular.module('userFactory', []);

UserFactory.factory('User', function (Alerts) {
    var _user = {name: "Barry Allen", coins: 333, debt: 0, message: ""};

    return {
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
                   "You made: " + balance: "You owe: " + balance;

            /*
            var res = 0;

            if ((_user.debt > _user.coins) && (_user.coins > 0)) {
                _user.message = "Congrats, you made a profit";
                Alerts.addAlert(_user.message);
                res = _user.coins;
            } else {
                _user.message = "You failed, and now owe money";
                Alerts.addAlert(_user.message, "error");
                res = -_user.coins;
            }

            return res*/
        },
        getMsg: function () {
            var balance = _user.debt - _user.coins;
            return balance > _user.debt && (_user.coins > 0) ?
                   "You owe: " + balance : "You made: " + balance;
            //return _user.message;
        }
    };
});
