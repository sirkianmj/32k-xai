[Setup]
SourceDir=..\..
AppName=32k-XAI
AppVersion=2.0
AppPublisher=Kian Mansouri Jamshidi & Sabihe Khorshidi
AppPublisherURL=https://github.com/sirkianmj/32k-xai
DefaultDirName={autopf}\32k-XAI
DefaultGroupName=32k-XAI
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=XAI-32k-Setup
SetupIconFile=packaging\windows\icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\XAI-32k.exe
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\XAI-32k\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\32k-XAI"; Filename: "{app}\XAI-32k.exe"
Name: "{autodesktop}\32k-XAI"; Filename: "{app}\XAI-32k.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\XAI-32k.exe"; Description: "{cm:LaunchProgram,32k-XAI}"; Flags: nowait postinstall skipifsilent
