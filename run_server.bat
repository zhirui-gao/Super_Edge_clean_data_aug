CALL :Main %* 3>>%0
GOTO :Eof
:Main

call activate pytorch3.6
call python E:\pytorch_model\Super_Edge_clean\minshi_predict.py

:Eof