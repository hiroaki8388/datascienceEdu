# 簡単な使い方
# test**.pyの名前の走査し、その中のassertをすべてテストする
def func(x):
    return x+1


# test_**で始まる関数をすべてテストする
def test_answer():
    assert func(4)==6