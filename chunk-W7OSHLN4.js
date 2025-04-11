import{a as _}from"./chunk-3BK4WEZJ.js";import{o as p}from"./chunk-GQDKQDED.js";import{$a as S,Ab as t,Cc as g,Pb as e,Sb as h,T as c,Tb as E,Wa as d,gb as u,hb as f,kb as r,vb as l,zb as n}from"./chunk-JUNAYWEM.js";import"./chunk-6MDQTQU3.js";function T(i,a){i&1&&(n(0,"p",2),e(1," Though this guide is for the Ubuntu linux; it should be easy to retrofit to others. "),t())}function R(i,a){i&1&&(n(0,"p",2),e(1," Though this guide has been tested on macOS High Sierra; it will probably work on older versions. "),t())}function k(i,a){i&1&&(n(0,"p",2),e(1," Follow this guide on Windows 7+. "),t())}function b(i,a){i&1&&e(0," Command Prompt (cmd.exe) ")}function G(i,a){i&1&&e(0," Terminal ")}function v(i,a){i&1&&(n(0,"pre",4)(1,"code",7),e(2,`
    sudo apt update
    sudo apt install build-essential git-core tcl
  `),t()())}function P(i,a){i&1&&(n(0,"a",19),e(1,"Latest XCode and Command Line Tools"),t(),n(2,"pre",4)(3,"code",7),e(4,`
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    brew update
    brew install git
  `),t()())}function y(i,a){i&1&&(n(0,"section",5),e(1,' Ensure you check "Add to PATH" whenever asked: '),n(2,"ol")(3,"li")(4,"a",20),e(5," Build Tools for Visual Studio 2017 "),t()(),n(6,"li")(7,"a",21),e(8,"git"),t()()(),n(9,"p",2),e(10,"Test that it installed correctly with:"),t(),n(11,"pre",4)(12,"code",7),e(13,`
      git --version
      cl /?
    `),t()()())}function A(i,a){i&1&&(n(0,"pre",4)(1,"code",7),e(2,`
    curl -L https://git.io/n-install | bash -s -- -y lts
  `),t()())}function D(i,a){i&1&&(n(0,"p",2),e(1," Download and setup from latest: "),n(2,"a",22),e(3,"'Windows Installer (.msi)' [LTS]"),t(),e(4,". "),t())}function O(i,a){i&1&&(n(0,"pre",4)(1,"code",7),e(2,`
    # Alternatively build from source. Quick guide: https://askubuntu.com/a/868862
    sudo add-apt-repository ppa:chris-lea/redis-server
    sudo apt update
    sudo apt install redis-server
  `),t()())}function M(i,a){i&1&&(n(0,"pre",4)(1,"code",7),e(2,`
    brew install redis
    brew services start redis
  `),t()())}function I(i,a){i&1&&(n(0,"p",2),e(1," Download and install the "),n(2,"a",23),e(3,"latest .msi"),t(),e(4,". "),t())}function L(i,a){i&1&&(n(0,"p",2),e(1," Follow "),n(2,"a",24),e(3,"their official guide"),t(),e(4,". I prefer using latest version from their apt repository. "),t())}function B(i,a){i&1&&(n(0,"p",2),e(1," Follow "),n(2,"a",25),e(3,"their official guide"),t(),e(4,". I prefer the "),n(5,"a",26),e(6,"EnterpriseDB package"),t(),e(7,". "),t())}function N(i,a){i&1&&(n(0,"p",2),e(1," Run this in an Administrator Command Prompt ("),n(2,"a",27),e(3,"how-to"),t(),e(4,"): "),t())}function U(i,a){i&1&&(n(0,"pre",4)(1,"code",7),e(2,`
    POSTGRES_DB_USER='REPLACE_ME'
    REST_PASS='REPLACE_ME' # recommend using \`read -s REST_PASS\`
    POSTGRES_DB_NAME='REPLACE_ME'

    createuser --superuser "$POSTGRES_DB_USER"
    psql -c "CREATE USER $POSTGRES_DB_USER WITH LOGIN PASSWORD '$REST_PASS';"
    createdb "$POSTGRES_DB_NAME" --owner "$POSTGRES_DB_USER"
    export RDBMS_URI="postgres://$POSTGRES_DB_USER:$REST_PASS@localhost/$POSTGRES_DB_NAME"
    # ^Recommend adding this \`export\` line to your ~/.bash_profile
  `),t()())}function W(i,a){i&1&&(n(0,"pre",4)(1,"code",28),e(2,`
    set POSTGRES_DB_USER="REPLACE_ME"
    set REST_PASS="REPLACE_ME"
    set POSTGRES_DB_NAME="REPLACE_ME"

    createuser --superuser "%POSTGRES_DB_USER%"
    psql -c "CREATE USER %POSTGRES_DB_USER% WITH LOGIN PASSWORD '%REST_PASS%';"
    createdb "%POSTGRES_DB_NAME%" --owner "%POSTGRES_DB_USER%"

    setx RDBMS_URI "postgres://%POSTGRES_DB_USER%:%REST_PASS%@localhost/%POSTGRES_DB_NAME%"
    :: This last line will add RDBMS_URI to your environment
  `),t()())}function j(i,a){i&1&&(n(0,"p"),e(1," Alternatively use "),n(2,"a",29),e(3,"my script"),t(),e(4,". "),t())}function $(i,a){i&1&&(n(0,"pre",4)(1,"code",7),e(2,`
    psql "$RDBMS_URI" -c 'SELECT 1'
  `),t()())}function q(i,a){i&1&&(n(0,"pre",4)(1,"code",7),e(2,`
    psql "%RDBMS_URI%" -c "SELECT 1"
  `),t()())}function F(i,a){i&1&&(n(0,"span")(1,"a",30)(2,"code"),e(3,"sed"),t()(),e(4," and "),t())}function H(i,a){i&1&&(n(0,"code"),e(1,"brew install hub"),t())}function Y(i,a){i&1&&(n(0,"span")(1,"a",31),e(2,"latest package"),t(),e(3," for your platform."),t())}function V(i,a){i&1&&(n(0,"pre",4)(1,"code",7),e(2),t()()),i&2&&(d(2),h(`
    hub create organisation/"$`,"{","PWD##*/","}",`" -d 'Description here'
  `))}var m=class i{constructor(a){this.platformPickerService=a;this.os=a.is.bind(a)}os;static \u0275fac=function(s){return new(s||i)(S(_))};static \u0275cmp=u({type:i,selectors:[["app-getting-started"]],standalone:!1,decls:104,vars:15,consts:[[1,"docs-markdown","pad"],[1,"docs-markdown-h5"],[1,"docs-markdown-p"],["id","step-0-install-build-dependencies",1,"docs-header-link","docs-markdown-h3"],[1,"docs-markdown-pre"],[1,"docs-guide-content"],["id","step-1-install-nodejs",1,"docs-header-link","docs-markdown-h3"],[1,"lang-bash","docs-markdown-code"],["id","step-2-install-redis",1,"docs-header-link","docs-markdown-h3"],["id","step-3-install-postgres",1,"docs-header-link","docs-markdown-h3"],["id","step-4-install-android",1,"docs-header-link","docs-markdown-h3"],["href","https://developer.android.com/studio/install.html",1,"docs-markdown-a"],["id","step-5-install-global-nodejs-packages",1,"docs-header-link","docs-markdown-h3"],["id","step-6-development-backend",1,"docs-header-link","docs-markdown-h3"],["id","step-7-angular-frontend",1,"docs-header-link","docs-markdown-h3"],["id","step-8-android",1,"docs-header-link","docs-markdown-h3"],["id","step-9-rebranding",1,"docs-header-link","docs-markdown-h3"],["href","https://github.com/sharkdp/fd"],["href","https://hub.github.com"],["href","https://developer.apple.com/downloads"],["href","https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15",1,"docs-markdown-a"],["href","https://git-scm.com/download/win",1,"docs-markdown-a"],["href","https://nodejs.org/en/download/",1,"docs-markdown-a"],["href","https://github.com/MicrosoftArchive/redis/releases",1,"docs-markdown-a"],["href","https://www.postgresql.org/download/linux/ubuntu/",1,"docs-markdown-a"],["href","https://www.postgresql.org/download/macosx/",1,"docs-markdown-a"],[1,"http://www.enterprisedb.com/products/pgdownload.do#macosx"],["href","https://technet.microsoft.com/en-us/library/cc947813(v=ws.10).aspx",1,"docs-markdown-a"],[1,"lang-batch","docs-markdown-code"],["href","https://github.com/offscale/offsh-postgres",1,"docs-markdown-a"],["href","https://chocolatey.org/packages/sed"],["href","https://github.com/github/hub/releases"]],template:function(s,o){s&1&&(n(0,"div",0)(1,"h5",1),e(2," This guides you to setting up your computer to run all my scaffolds. "),t(),r(3,T,2,0,"p",2)(4,R,2,0,"p",2)(5,k,2,0,"p",2),n(6,"em"),e(7," All grey background assumes you are running in an active "),r(8,b,1,0)(9,G,1,0),e(10," session "),t(),n(11,"h3",3),e(12," Step 0: Install build dependencies "),t(),r(13,v,3,0,"pre",4)(14,P,5,0)(15,y,14,0,"section",5),n(16,"h3",6),e(17," Step 1: Install Node.js "),t(),r(18,A,3,0,"pre",4)(19,D,5,0,"p",2),n(20,"p",2),e(21,"Test that it installed correctly with:"),t(),n(22,"pre",4)(23,"code",7),e(24,`
    node --version
    npm --version
  `),t()(),n(25,"h3",8),e(26," Step 2: Install Redis "),t(),r(27,O,3,0,"pre",4)(28,M,3,0,"pre",4)(29,I,5,0,"p",2),n(30,"p",2),e(31,"You can test it's installed and running with:"),t(),n(32,"pre",4)(33,"code",7),e(34,`
    redis-cli ping
  `),t()(),n(35,"p",2),e(36,"If that failed, open a new session and run this, then try the ping again:"),t(),n(37,"pre",4)(38,"code",7),e(39,`
    redis-server
  `),t()(),n(40,"h3",9),e(41," Step 3: Install and configure Postgres "),t(),r(42,L,5,0,"p",2)(43,B,8,0,"p",2)(44,N,5,0,"p",2)(45,U,3,0,"pre",4)(46,W,3,0,"pre",4)(47,j,5,0,"p"),n(48,"p",2),e(49,"You can test it's installed and running with:"),t(),r(50,$,3,0,"pre",4)(51,q,3,0,"pre",4),n(52,"h3",10),e(53," Step 4: Install Android dependencies and IDE "),t(),n(54,"p",2),e(55," See "),n(56,"a",11),e(57,"official install guide"),t(),e(58,". "),t(),n(59,"h3",12),e(60," Step 5: Install global Node.js packages "),t(),n(61,"pre",4)(62,"code",7),e(63,`
    npm i -g bunyan typings typescript @angular/cli
  `),t()(),n(64,"h3",13),e(65," Step 6: Development backend (Node.js) "),t(),n(66,"pre",4)(67,"code",7),e(68,`
    git clone https://github.com/SamuelMarks/restify-orm-scaffold
    cd restify-orm-scaffold
    typings i
    tsc
    npm start
  `),t()(),n(69,"h3",14),e(70," Step 7: Angular frontend (web) "),t(),n(71,"pre",4)(72,"code",7),e(73,`
    git clone https://github.com/SamuelMarks/ng-material-scaffold
    cd ng-material-scaffold
    npm i
    npm start
  `),t()(),n(74,"h3",15),e(75," Step 8: Android "),t(),n(76,"pre",4)(77,"code",7),e(78,`
    git clone https://github.com/SamuelMarks/android-auth-scaffold
  `),t()(),n(79,"p",2),e(80," Now open that in Android Studio; build and run it. "),t(),n(81,"h3",16),e(82," Step 9: Branding "),t(),n(83,"p",2),e(84,"Now is a good time to rename everything. Close all open windows, and let's begin."),t(),n(85,"p",2),e(86,"Install "),r(87,F,5,0,"span"),n(88,"a",17)(89,"code"),e(90,"fd"),t()(),e(91,". Then within each of the repo directories, or from a parent directory, run:"),t(),n(92,"pre",4)(93,"code",7),e(94),t()(),n(95,"p",2),e(96,"You will need likely want to update descriptions and add DVCS repositories also. Install "),n(97,"a",18)(98,"code"),e(99,"hub"),t()(),e(100," with "),r(101,H,2,0,"code")(102,Y,4,0,"span"),t(),r(103,V,3,2,"pre",4),t()),s&2&&(d(3),l(o.os("Linux")?3:o.os("macOS")?4:o.os("Windows")?5:-1),d(5),l(o.os("Windows")?8:o.os("macOS")||o.os("Linux")?9:-1),d(5),l(o.os("Linux")?13:o.os("macOS")?14:o.os("Windows")?15:-1),d(5),l(o.os("Linux")||o.os("macOS")?18:o.os("Windows")?19:-1),d(9),l(o.os("Linux")?27:o.os("macOS")?28:o.os("Windows")?29:-1),d(15),l(o.os("Linux")?42:o.os("macOS")?43:o.os("Windows")?44:-1),d(3),l(o.os("Linux")||o.os("macOS")?45:o.os("Windows")?46:-1),d(2),l(o.os("Linux")||o.os("macOS")?47:-1),d(3),l(o.os("Linux")||o.os("macOS")?50:o.os("Windows")?51:-1),d(37),l(o.os("Windows")?87:-1),d(7),E(`
    fd -t f -exec sed -i 's/ng-material-scaffold/projectname/g' `,"{}",` \\;
    fd -t f -exec sed -i 's/NgMaterialScaffold/projectname/g' `,"{}",` \\;

    fd -t f -exec sed -i 's/restify-orm-scaffold/projectname/g' `,"{}",` \\;
  `),d(7),l(o.os("macOS")?101:o.os("Linux")?102:-1),d(2),l(o.os("Linux")||o.os("macOS")?103:-1))},styles:[".pad[_ngcontent-%COMP%]{padding:0 3em}"]})};var x=[{path:"",component:m}];var w=class i{static \u0275fac=function(s){return new(s||i)};static \u0275mod=f({type:i});static \u0275inj=c({imports:[g,p,p.forChild(x)]})};export{w as GettingStartedModule};
