import sys
current_module=sys.modules[__name__]

# mock

# monkeypatch.setattr(a, b, c)
# a:モックにしたい[関数, クラス]を含む[モジュール, クラス]
# b:モックにしたい[関数名, クラス名]
# c:モックの[関数, クラス]

# 関数をmock
def bar():
    return "bar"

def test_bar(monkeypatch):
    monkeypatch.setattr(current_module,'bar',lambda:'patch')
    assert bar()=='patch'
    
# class内の関数をmock
class Piyo(object):
    def piyo_func(self):
        return 'piyo!'
 

def hoge():
    piyo=Piyo()
    
    return piyo.piyo_func()

def test_piyo(monkeypatch):
    monkeypatch.setattr(current_module.Piyo,'piyo_func',lambda *args:'patch')
    
    assert hoge() == 'patch' 