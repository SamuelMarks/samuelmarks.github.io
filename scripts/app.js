'use strict';

var ModalInstanceCtrl = function($scope, $modalInstance, $log, $templateCache, $location, items, User) {
    $scope.items = items;
    $scope.exit_market = User.exitMarket;

    $scope.selected = {
        item: $scope.items[0]
    };

    $scope.ok = function() {
        $modalInstance.close($scope.selected.item);
    };

    $scope.cancel = function() {
        $modalInstance.dismiss('cancel');
    };

    $scope.cleanSlate = function() {
        $templateCache.removeAll();
        User.toDefault();
    };

    $scope.restart = function() {
        $scope.cleanSlate();

        $location.path("/");

        $scope.ok();
    };
};

/**
 * @ngdoc overview
 * @name mas340App
 * @description
 * # mas340App
 *
 * Main module of the application.
 */
var mas340App = angular.module(
    'mas340App', ['mainCtrl', 'titlebarCtrl', 'embarkCtrl', 'alertsCtrl', 'userCtrl', 'enterCtrl',
                  'alertsFactory', 'userFactory',
                  'ui.bootstrap', 'ui.bootstrap.tpls', 'ui.bootstrap.transition', 'ui.router',
                  'ngAnimate', 'ngResource', 'ngTouch', 'ngSanitize']
);

// routes / states
mas340App.config(
    function($stateProvider, $urlRouterProvider) {
        $urlRouterProvider.otherwise("/");

        $stateProvider
            .state('index', {
                       url: "/",
                       templateUrl: "/samuelmarks.github.io/views/home.html",
                       controller: 'MainCtrl'
                   })
            .state('alerts', {
                       url: '/alerts'
                   })
            .state('sidebar', {
                       url: '/sidebar',
                       templateUrl: '/samuelmarks.github.io/views/sidebar.html',
                       controller: 'UserCtrl'
                   })
            .state('embark', {
                       url: '/embark',
                       views: {
                           '': { templateUrl: '/samuelmarks.github.io/views/embark.html', controller: 'EmbarkCtrl'},
                           'sidebar@embark': { templateUrl: '/samuelmarks.github.io/views/sidebar.html', controller: 'UserCtrl' }
                       }
                   })
            .state('embark.stories', {
                       abstract: true,
                       views: {
                           /*"colorsMenu": {
                            template: '<div class="col-sm-6">' +
                            '<h5>Pick a story:</h5>' +
                            '<ul>' +
                            '<li><a ui-sref="embark.stories.lemonade">Lemonade</a></li>' +
                            '<li><a ui-sref="embark.stories.smoothies">Smoothies</a></li>' +
                            '<li><a ui-sref="embark.stories.donuts">Donuts</a></li>' +
                            '</ul>' +
                            '</div>'
                            },*/
                           "": {
                               template: '<div class="col-sm-12">' +
                                         '<div ui-view></div>' +
                                         '</div>'
                           }
                       }
                   })
            .state('embark.stories.lemonade', {
                       url: '/stories/lemonade',
                       templateUrl: '/samuelmarks.github.io/views/story/0.html',
                       controller: function($scope) {
                           var scopeVal = "Lemonade";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal)
                       }
                   })
            .state('embark.stories.enter', {
                       url: '/enter',
                       templateUrl: '/samuelmarks.github.io/views/story/enter.html',
                       controller: 'EnterCtrl'
                   })
            .state('embark.stories.exit', {
                       url: '/exit',
                       templateUrl: '/samuelmarks.github.io/views/story/exit.html',
                       controller: function($scope) {
                           var scopeVal = "Exit";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal)
                       }
                   })
            .state('embark.stories.smoothie', {
                       url: '/stories/smoothie',
                       templateUrl: '/samuelmarks.github.io/views/story/fail_1.html',
                       controller: function($scope) {
                           var scopeVal = "Bad choice";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal)
                       }
                   })
            .state('embark.stories.donuts', {
                       url: '/stories/donuts',
                       templateUrl: '/samuelmarks.github.io/views/story/fail_2.html',
                       controller: function($scope) {
                           var scopeVal = "Wrong choice";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal)
                       }
                   });
    }
);

mas340App.config(function($logProvider) {
    $logProvider.debugEnabled(true);
});

mas340App.config(function($locationProvider) {
    $locationProvider.html5Mode(true).hashPrefix('/');
});
