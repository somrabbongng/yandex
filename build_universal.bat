@echo off
echo Starting build process...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Clean up any existing virtual environment
if exist "venv" (
    echo Removing existing virtual environment...
    rmdir /s /q "venv"
)

REM Create fresh virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Fix pip by downloading get-pip.py and reinstalling
echo Fixing pip installation...
powershell -Command "& { Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py' }"
python get-pip.py --force-reinstall
del get-pip.py

REM Install required packages
echo Installing requirements...
python -m pip install --upgrade pip
python -m pip install pyinstaller
python -m pip install PyQt6
python -m pip install fastapi
python -m pip install uvicorn
python -m pip install pydantic
python -m pip install selenium
python -m pip install webdriver-manager
python -m pip install cachetools
python -m pip install aiohttp
python -m pip install certifi
python -m pip install urllib3
python -m pip install requests

REM Create dist directory if it doesn't exist
if not exist "dist" mkdir dist

REM Download Visual C++ Redistributable
echo Downloading Visual C++ Redistributable...
powershell -Command "& { Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile 'dist\vc_redist.x64.exe' }"

REM Build executable using PyInstaller
echo Building executable...
pyinstaller --noconfirm --clean ^
    --name "UniversalMailClient" ^
    --add-data "venv\Lib\site-packages\selenium\webdriver\common\windows\selenium-manager.exe;selenium\webdriver\common\windows" ^
    --add-data "venv\Lib\site-packages\selenium\webdriver\common\macos\selenium-manager;selenium\webdriver\common\macos" ^
    --add-data "venv\Lib\site-packages\selenium\webdriver\common\linux\selenium-manager;selenium\webdriver\common\linux" ^
    --hidden-import "PyQt6.sip" ^
    --hidden-import "PyQt6.QtCore" ^
    --hidden-import "PyQt6.QtGui" ^
    --hidden-import "PyQt6.QtWidgets" ^
    --hidden-import "pydantic" ^
    --hidden-import "uvicorn.logging" ^
    --hidden-import "uvicorn.loops" ^
    --hidden-import "uvicorn.loops.auto" ^
    --hidden-import "uvicorn.protocols" ^
    --hidden-import "uvicorn.protocols.http" ^
    --hidden-import "uvicorn.protocols.http.auto" ^
    --hidden-import "uvicorn.protocols.websockets" ^
    --hidden-import "uvicorn.protocols.websockets.auto" ^
    --hidden-import "uvicorn.lifespan" ^
    --hidden-import "uvicorn.lifespan.on" ^
    --hidden-import "uvicorn.main" ^
    --hidden-import "fastapi" ^
    --hidden-import "cachetools" ^
    --hidden-import "aiohttp" ^
    --hidden-import "certifi" ^
    --hidden-import "urllib3" ^
    --hidden-import "requests" ^
    --hidden-import "threading" ^
    --noconsole ^
    --onefile ^
    universal.py

REM Create requirements.txt
echo Creating requirements.txt...
(
echo PyQt6
echo fastapi
echo uvicorn
echo pydantic
echo selenium
echo webdriver-manager
echo cachetools
echo aiohttp
echo certifi
echo urllib3
echo requests
) > dist\requirements.txt

REM Create readme file
echo Creating readme file...
(
echo UniversalMailClient
echo ===============
echo.
echo Pre-Installation Requirements:
echo 1. Install Visual C++ Redistributable 2015-2022:
echo    - Run vc_redist.x64.exe included in this package
echo    - Or download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo Installation Instructions:
echo 1. Run vc_redist.x64.exe if not already installed
echo 2. Double-click UniversalMailClient.exe to run the application
echo 3. The application will automatically manage Chrome WebDriver
echo.
echo Note: Make sure you have Google Chrome installed on your system.
) > dist\readme.txt

echo.
echo Build process completed successfully!
echo The executable can be found in the dist folder.
pause