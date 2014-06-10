'use strict';

var UserFactory = angular.module('userFactory', []);

UserFactory.factory('User', function () {
    var _user = {name: "Barry Allen", coins: 0.0, badges: []};

    return {
        user: function () {
            return _user;
        },
        set: function (key, value) {
            _user.key = value;
        }
    };
});
