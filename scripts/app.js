'use strict';

/**
 * @ngdoc overview
 * @name mas340App
 * @description
 * # mas340App
 *
 * Main module of the application.
 */
var mas340App = angular.module(
    'mas340App', ['mainCtrl', 'titlebarCtrl', 'embarkCtrl', 'alertsCtrl', 'userCtrl',
                  'alertsFactory', 'userFactory',
                  'ui.bootstrap', 'ui.bootstrap.tpls', 'ui.bootstrap.transition',
                  'ngRoute', 'ngAnimate', 'ngResource', 'ngTouch']
);

// routes / states

mas340App.config(
    ['$routeProvider', function ($routeProvider) {
        $routeProvider
            .when('/', {
                      templateUrl: '/app/views/home.html',
                      controller: 'MainCtrl'
                  })
            .when('/embark', {
                      templateUrl: '/app/views/embark.html',
                      controller: 'EmbarkCtrl'
                  })
            .when('/alerts', {
                      templateUrl: '/app/views/alerts.html',
                      controller: 'AlertsCtrl'
                  })
            .when('/sidebar', {
                      templateUrl: '/app/views/sidebar.html',
                      controller: 'UserCtrl'
                  })
            .otherwise({redirectTo: '/'});
    }]
);

mas340App.config(function ($logProvider) {
    $logProvider.debugEnabled(true);
});

mas340App.config(function ($locationProvider) {
    $locationProvider.html5Mode(true).hashPrefix('/');
});
