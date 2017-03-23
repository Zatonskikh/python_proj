import math
import sympy
import numpy
def Frequency_Monobits_Test (data):
    amount=0
    count=0
    if data == 0:
        amount=count=1
    while data != 0:
        a = data & 1
        if a == 0:
            a = -1
        data = data >> 1
        count+=1
        amount += a
    
    static = abs(amount)/math.sqrt(count)/math.sqrt(2)
    
    #erfc = (2/math.sqrt(math.pi))*(sympy.integrate(sympy.exp(-sympy.Symbol('x')**2),(sympy.Symbol('x'),static,sympy.Symbol('oo'))))
    # return math.erfc(static)

def Compare(value):
    p = 0.01
    if  value < p:
        print '������������������ �� �������� ���������.'
    elif value == 0xff :
        print '���� �� ����� ���� ��������.'
    else:
        print '��� ��.'  



def Frequency_Block_Test (data, data_len):
    blocks = []
    s = []
    mult = 1
    x = 0
    while data_len != 0:
        if data_len < 3:
            data <<= (3 - data_len)
            data_len = 3
            
        blocks.append(data >> (data_len - 3))
        data_len -= 3
        if data_len==1 and (data==0 or data==1):
            mult=1
        else:
            while x < data_len-1:
                mult = mult << 1
                mult += 1
                x += 1
            data = data & mult
            x,mult=0,1
    if blocks[len(blocks) - 1] == 0:
        del blocks[len(blocks) - 1]
    
    
    for item in blocks:
        s.append ((float(((item&0b1)+((item&0b10)>>0b1)+((item&0b100)>>0b10)))/3-0.5)**2)
    #print 12*(sum (s))
    #print math.gamma(float(len(blocks))/2)
    return sympy.uppergamma(float(len(blocks))/2,(sum (s))*6)
def Same_In_Row_Test(data, data_len):
        temp_data = data
        count_1 = 0
        arg = 0
        p = 0
        v = 0
       
        while (data != 0):
                if data % 2 == 1:
                        count_1 += 1
                data /= 2
       
        p = float(count_1) / data_len
       
        if abs(p - 0.5) < 2 / math.sqrt(data_len):
                while temp_data != 1:
                        if temp_data % 2 != (temp_data % 4)>>1:
                                v += 1
                                
                        temp_data /= 2
        v += 1
        try:
            arg = abs(v - 2*data_len*p*(1 - p))/(2*math.sqrt(2*data_len)*p*(1-p))
            result = math.erfc(arg)
        except ZeroDivisionError:
            result = 0xff     
        return result
def Test_For_The_Longest_Run_Of_Ones (data, data_len):
    if data_len <= 128:
        M, K, R = 8, 3, 16
        blocks_len4stat = [1, 2, 3, 4]
        p = [0.2148, 0.3672, 0.2305, 0.1875]
    elif data_len <= 6272:
        M, K, R = 128, 5, 49
        blocks_len4stat = [4, 5, 6, 7, 8, 9]
        p = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, K, R = 10000, 6, 75
        blocks_len4stat = [10, 11, 12, 13, 14, 15, 16]
        p = [0.0228, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
   
    # Разбиение последовательности на блоки
    blocks = []
    count_seq_len = []
    ind = 0
    while data_len > 0:
        blocks.append(str(bin((data % ( 2 ** M)))))
        data /= 2 ** M
        data_len -= M
        if (len(blocks[ind]) != M + 2):
            temp = len(blocks[ind])
            for i in range (M + 2 - temp):
                blocks[ind] += '0'
        ind += 1
 
    # Вычисление максимальной последовательности единиц для каждого блока
    length = 0
   
    for j in range(0, len(blocks)):
        length_max = 0
       
        for i in range(0, M+2):      
            if blocks[j][i] == '1':
                length += 1
 
            if blocks[j][i] == '0' or i == M+1:
                if length > length_max:
                    length_max = length
                length = 0
 
        # Добавление длины последовательности в list
        count_seq_len.append(length_max)
 
    # Вычисление статистики
    stat = [0, 0, 0, 0, 0, 0, 0]
   
    for i in range(len(blocks_len4stat)):
        for item in count_seq_len:
            if item < blocks_len4stat[0]:
                stat[0] += 1
            elif item == blocks_len4stat[i]:
                stat[i] += 1
            elif item > blocks_len4stat[len(blocks_len4stat) - 1]:
                stat[len(blocks_len4stat) - 1] += 1
               
    # Вычисление Хи-квадрата
    xi = 0
    for i in range(0, K + 1):
        xi += ((stat[i] - R * p[i]) ** 2) / (R * p[i])
 
    result = sympy.uppergamma(float(K)/2, float(xi)/2)
    return result
def Binary_Matrix_Rank_Test (data, data_len):
    M = Q = 3
    # Число матриц
    N = data_len / (M * Q)
    pm = [0.2888, 0.5776, 0.1284, 0.005, 0]
   
    matrix = []
    matrix_rank = []
    fm = []
 
    data = data >> (data_len % (M * Q))
    for i in range(N):
        array = []
        x = data % (2 ** (M * Q))
        data >>= (M * Q)
        data_len -= M * Q
 
        for j in range (M * Q):
            array.append(x % 2)
            x /= 2
        matrix.append(sympy.Matrix(M, Q, array))
 
        matrix_rank.append(matrix[i].rank())
 
    for i in range(M + 1):
        fm.append(0)
 
    for rank in matrix_rank:
        fm[rank] += 1
 
    p = 0.005
    for i in range(0, M + 1):
        if i == M:
            p = pm[0]
        elif i == M - 1:
            p = pm[1]
        elif i == M - 2:
            p = pm[2]
        elif i == M - 3:
            p = pm[3]
           
        fm[i] = float(((fm[i] - N * p) ** 2)) / (N * p)
       
    xi = 0
    xi = sum(fm)
 
    result = sympy.uppergamma(1, float(xi)/2)
    return result
def Spectral_Test(data, data_len):
    run =[]
    run2=[]
    four=[]
    while data != 0:
        a = data & 1
        if a == 0:
            a = -1
        data = data >> 1
        run.append(a)
    run2 = run[::-1]
    four = abs(numpy.fft.fft(run2))
    T=math.sqrt((math.log(1/0.05))*data_len)
    
		
    N0=0.95*datalen/2
    N1=math.floor(N0)
    d=float(N1-N0)/math.sqrt(float(data_len*0.95*0.05)/4)
    result = sympy.erfc(float(abs(d))/math.sqrt(2))
    
    for item in range(data_len/2):
        if four[item]>T:
            result=0xff
    return result
	
#DIEHARD

def overlapping_permutation_test(data,data_len):
    blocks = []
    s = []
    mult = 1
    x= Xi = 0
    v1,v2,v3,v4,v5,v6=0,0,0,0,0,0
    V = [0,0,0,0,0,0]
    c=0
    
    data>>=(data_len%3)
    data_len-=data_len%3
    n = data_len/3
    N = (n-1)/2
    N1 = float(N)/6
    while data_len!=0:
        
        blocks.append(data >> (data_len - 3))
        data_len -= 3
        
        while x < data_len-1:
            mult = mult << 1
            mult += 1
            x += 1
        data = data & mult
        x,mult=0,1
    for i in range(0,len(blocks)-3,2):
        if blocks[i]==blocks[i+1] or blocks[i]==blocks[i+2] or blocks[i+2]==blocks[i+1]:
            c+=1
        else:
            if blocks[i]<blocks[i+1]:
                if blocks[i+1]<blocks[i+2]:
                    V[0]+=1
                else:
                    if blocks[i]<blocks[i+2]:
                        V[1]+=1
                    else:
                        V[4]+=1
            else :
                if blocks[i]<blocks[i+2]:
                    V[2]+=1
                else:
                    if blocks[i+1]<blocks[i+2]:
                        V[3]+=1
                    else :
                        V[5]+=1
                    
    for i in range(0,6):
        Xi+=float(((V[i]-N1)**2))/N1
   
    return sympy.uppergamma(2.5, Xi/2)/sympy.gamma(2.5)
#�����
def coupones_test(data,data_len):
        blocks = []
        Vi = [0]*3
        s=[0]*4
        mult = 1
        x= Xi = 0
        d=4
        len_one=0
        data>>=(data_len%2)
        data_len-=data_len%2
        while data_len!=0:
        
                        blocks.append(data >> (data_len - 2))
                        data_len -= 2
        
                        while x < data_len-1:
                                mult = mult << 1
                                mult += 1
                                x += 1
                        data = data & mult
                        x,mult=0,1
        
        for i in range(len(blocks)):
                if blocks[i]==0:
                        s[0]+=1
                        len_one+=1
                elif blocks[i]==1:
                        s[1]+=1
                        len_one+=1
                elif blocks[i]==2:
                        s[2]+=1
                        len_one+=1
                else :
                        s[3]+=1
                        len_one+=1
                if reduce(lambda res, x: res*x, s, 1)>0:
                        if len_one==4:Vi[0]+=1
                        elif len_one==5:Vi[1]+=1
                        else: Vi[2]+=1
                        for j in range(4):s[j]=0
                        len_one=0
        for i in range(3):
                if i<2:
                        p=float(math.factorial(d))/(d**(i+4))*sympy.mpmath.stirling2(i+3,d-1)
                        
                else:
                        p=1-float(math.factorial(d))/(d**5)*sympy.mpmath.stirling2(5,d)
                        
                Xi+=((Vi[i]-(sum(Vi)*p))**2)/(sum(Vi)*p)
                
                
        return sympy.uppergamma(2, Xi/2)/sympy.gamma(2)    


   
        
	
        
    
    
seq = input("Enter data:")
datalen = input("Enter len:")              
temp = Frequency_Monobits_Test(seq)
#print temp
Compare(temp)    
temp=Frequency_Block_Test(seq, datalen)
Compare(temp)
temp=Same_In_Row_Test(seq, datalen)
Compare(temp)
temp=Test_For_The_Longest_Run_Of_Ones (seq, datalen)
Compare(temp)
temp=Binary_Matrix_Rank_Test(seq, datalen)
Compare(temp)
temp=Spectral_Test(seq, datalen)
Compare(temp)
overlapping_permutation_test(seq,datalen)
        
        
        

