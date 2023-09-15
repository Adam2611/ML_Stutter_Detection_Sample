#this file tests all the functions in preprocessing. run test() to see if all tests pass

import preprocessing_functions as func

def test_all():
    test_padding()
    test_alter_spike()
    test_shuffle()
    test_normalize_all()
    test_normalize_each()
    test_standardize_all()
    test_standardize_each()
    test_split()
    test_duplicate()
    print("\n----------------------------------------------")
    print("Passed All Test Cases")
    print("----------------------------------------------")

def test_padding():
    in1 = [[1,2,3,4,5,6,7,8,9,10], [11, 12, 13, 14, 15, 16]]
    out1 = func.padding_x(in1, 5)
    out2 = func.padding_x(in1, 12)

    result1 = []
    result1.append([1,2,3,4,5])
    result1.append([11,12,13,14,15])
    result2 = []
    result2.append([1,2,3,4,5,6,7,8,9,10,0,0])
    result2.append([11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0])

    assert (out1 == result1).all()
    assert (out2 == result2).all()

def test_alter_spike():
    in1 = [[1,2,3,4,5,6,7,8,9,100], [110, 120, 130, 140, 150, 160]]
    out1 = func.alter_spike(in1, 100, 2)
    
    result1 = []
    result1.append([1,2,3,4,5,6,7,8,9,200])
    result1.append([220, 240, 260, 280, 300, 320])

    assert (out1==result1)

def test_shuffle():
    list1 = [ [10,10,10], [59, 59, 59], [80, 80, 80]]
    list2 = [ [3,3,3], [51, 51, 51], [20,20,20]]
    combined=[list1, list2]
    num1 = list1[0][0] + list2[0][0]
    num2 = list1[1][1] + list2[1][1]
    num3 = list1[2][2] + list2[2][2]
    total = (num1+num2)+num3

    a,b = func.shuffle(list1, list2)


    out1 = a[0][0] + b[0][0]
    out2 = a[1][1] + b[1][1]
    out3 = a[2][2] + b[2][2]
    out_t = (out1+out2)+out3

    assert total==out_t

def test_normalize_all():
    in_train1 = [[1,2,3], [4,5,6], [7,8,9], [10,11]]
    in_train2 = [[0,5,10], [15,20]]
    out1, scaler = func.normalize_all(in_train1, 3)
    out2 = func.normalize_all(in_train2, 3, scaler)
    
    out3=[]
    out4=[]
    res1=[]
    res2=[]

    for x in out1:
        for x2 in x:
            out3.append(round(x2[0], 4))

    for y in out2:
        for y2 in y:
            out4.append(round(y2[0], 4))
    
    res1.append([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    res2.append([-0.1, 0.4, 0.9, 1.4, 1.9])

    assert (out3==res1[0])
    assert (out4==res2[0])
    assert (len(out1) == len(in_train1))
    assert (len(out2) == (len(in_train2)))

def test_normalize_each():
    in1 = [[1,2,3], [30,40,50,60,70]]
    out1 = func.normalize_each(in1)
    
    result1 = []
    result1.append([0.0, 0.5, 1.0])
    result1.append([0.0, 0.25, 0.5, 0.75, 1.0])

    assert (out1==result1)

def test_standardize_all():
    in_train1 = [[-2, -1], [0,1], [2,3], [4,5], [6,7]]
    in_train2 = [[2.5, 3.5], [4.5, 5.5]]
    out1, scaler = func.standardize_all(in_train1, 2)
    out2 = func.standardize_all(in_train2, 2, scaler)
 
    out3=[]
    out4=[]
    res1=[]
    res2=[]

    for x in out1:
        for x2 in x:
            out3.append(round(x2[0], 4))

    for y in out2:
        for y2 in y:
            out4.append(round(y2[0], 4))
    
    res1.append([-1.5667, -1.2185, -0.8704, -0.5222, -0.1741, 0.1741, 0.5222, 0.8704, 1.2185, 1.5667])
    res2.append([0.0, 0.3482, 0.6963, 1.0445])

    assert (out3==res1[0])
    assert (out4==res2[0])
    assert (len(out1) == len(in_train1))
    assert (len(out2) == (len(in_train2)))

def test_standardize_each():
    in1 = [[-1, 0, 1], [1,2,3,4]]
    out1 = func.standardize_each(in1)
    result1 = []
    result1.append([-1.224744871391589, 0.0, 1.224744871391589])
    result1.append([-1.3416407864998738, -0.4472135954999579, 0.4472135954999579, 1.3416407864998738])

    assert (out1==result1)

def test_split():
    inx1 = [[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15,16], [17,18], [19,20]]
    iny1 = [0,0,0,0,0,1,1,1,1,1]

    outx1, outx2, outx3, outy1, outy2, outy3 = func.split(inx1, iny1, 0.3, 0.2)

    res1 = [[1,2], [3,4], [5,6], [7,8], [9,10]]
    res2 = [[11,12], [13,14], [15,16]]
    res3 = [[17,18], [19,20]]
    res4 = [0,0,0,0,0]
    res5 = [1,1,1]
    res6 = [1,1]

    assert (res1==outx1)
    assert (res2==outx2)
    assert (res3==outx3)
    assert (res4==outy1)
    assert (res5==outy2)
    assert (res6==outy3)

def test_duplicate():
    inx1 = [[1,2], [3,4]]
    inx2 = [[11,12], [13,14], [15,16]]
    iny1 = [0,0,0]
    iny2 = [1,1,1,1,1]

    out1, out2, out3, out4 = func.duplicate(inx1, inx2, iny1, iny2, 2)
 
    res1 = [[1,2],[3,4],[1,2],[3,4]]
    res2 = [[11,12],[13,14],[15,16],[11,12],[13,14],[15,16]]
    res3 = [0,0,0,0,0,0]
    res4 = [1,1,1,1,1,1,1,1,1,1]

    assert out1==res1
    assert out2==res2
    assert out3==res3
    assert out4==res4

if __name__ == "__main__":
    test_all()
