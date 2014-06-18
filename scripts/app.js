'use strict';

var ModalInstanceCtrl = function ($scope, $modalInstance, items, User) {
    $scope.items = items;
    $scope.exit_market = User.exitMarket;
    $scope.get_msg = User.getMsg();

    $scope.selected = {
        item: $scope.items[0]
    };

    $scope.ok = function () {
        $modalInstance.close($scope.selected.item);
    };

    $scope.cancel = function () {
        $modalInstance.dismiss('cancel');
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
    'mas340App', ['mainCtrl', 'titlebarCtrl', 'embarkCtrl', 'alertsCtrl', 'userCtrl',
                  'alertsFactory', 'userFactory',
                  'ui.bootstrap', 'ui.bootstrap.tpls', 'ui.bootstrap.transition', 'ui.router',
                  'ngAnimate', 'ngResource', 'ngTouch', 'ngSanitize']
);

// routes / states
mas340App.config(
    function ($stateProvider, $urlRouterProvider) {
        $urlRouterProvider.otherwise("/");

        $stateProvider
            .state('index', {
                       url: "/",
                       templateUrl: "/views/home.html",
                       controller: 'MainCtrl'
                   })
            .state('alerts', {
                       url: '/alerts'
                   })
            .state('sidebar', {
                       url: '/sidebar',
                       templateUrl: '/views/sidebar.html',
                       controller: 'UserCtrl'
                   })
            .state('embark', {
                       url: '/embark',
                       views: {
                           '': { templateUrl: '/views/embark.html', controller: 'EmbarkCtrl'},
                           'sidebar@embark': { templateUrl: '/views/sidebar.html', controller: 'UserCtrl' }
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
                       templateUrl: '/views/story/0.html',
                       controller: function ($scope) {
                           var scopeVal = "Lemonade";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal)
                       }
                   })
            .state('embark.stories.enter', {
                       url: '/enter',
                       templateUrl: '/views/story/enter.html',
                       controller: function ($scope, $modal, $log, User) {
                           var scopeVal = "Enter";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal);
                           $scope.dice_roll = function () {
                               var satoshi = Math.floor(Math.random() * 1000);
                               if (Math.random() > 0.5) User.user().coins += satoshi;
                               else User.user().coins -= satoshi;
                           };

                           $scope.items = ['item1', 'item2', 'item3'];

                           $scope.open = function (size) {
                               var modalInstance = $modal.open(
                                   {
                                       templateUrl: '/views/story/exit.html',
                                       controller: ModalInstanceCtrl,
                                       size: size,
                                       resolve: {
                                           items: function () {
                                               return $scope.items;
                                           }
                                       }
                                   });

                               modalInstance.result.then(function (selectedItem) {
                                   $scope.selected = selectedItem;
                               }, function () {
                                   $log.info('Modal dismissed at: ' + new Date());
                               });
                           };
                       }
                   })
            .state('embark.stories.exit', {
                       url: '/exit',
                       templateUrl: '/views/story/exit.html',
                       controller: function ($scope) {
                           var scopeVal = "Exit";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal)
                       }
                   })
            .state('embark.stories.smoothie', {
                       url: '/stories/smoothie',
                       templateUrl: '/views/story/fail_1.html',
                       controller: function ($scope) {
                           var scopeVal = "Bad choice";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal)
                       }
                   })
            .state('embark.stories.donuts', {
                       url: '/stories/donuts',
                       templateUrl: '/views/story/fail_2.html',
                       controller: function ($scope) {
                           var scopeVal = "Wrong choice";
                           if ($scope.previous_states.indexOf(scopeVal) === -1)
                               $scope.previous_states.push(scopeVal)
                       }
                   });
    }
);

mas340App.config(function ($logProvider) {
    $logProvider.debugEnabled(true);
});

mas340App.config(function ($locationProvider) {
    $locationProvider.html5Mode(true).hashPrefix('/');
});
