from cx_Freeze import setup, Executable

setup(name="Translator model runner", executables=[Executable("Translator model runner script.py")], options={"build_exe": {"excludes": ["tkinter"]}})