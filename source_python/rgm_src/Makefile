#F77     =      ifort
#FFLAGS  =       -O2 -g -traceback -i8 -mcmodel=medium
#LIBS    = -L${MKLROOT} -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core  -liomp5  -lm -ldl

F77     =      gfortran
FFLAGS  =       -g -O2 -mcmodel=medium #-finteger-4-integer-8 #-fdefault-integer-8 -ffpe-trap=denormal 
LIBS    =  -lblas -llapack

#LIBS    =       -L/opt/f77/lib:/home/hmh/lib -leispack -lsu2
#LIBS    =       -L/home/hmh/lib -leispack -lsu2 -llapack -lblas
#LIBS    =       -L/home/hmh/lib -L/opt/intel_mkl_61/lib 
#LIBS    =       -L/home/hmh/lib -leispack -lsu2  -lblas

.f.o:
	$(F77) $(FFLAGS) -c -o $*.o $<


clean:
	rm -f *.o core

QUAFL_EFT_new: QUAFL_EFT_new.o par/QUAF_EFT
	$(F77) $(FFLAGS) -o QUAFL_DD.exe QUAFL_EFT_new.o 

LUDW_EFT_new: LUDW_EFT_new.o par/LUCN_EFT
	$(F77) $(FFLAGS) -o LUDW_EFT_new.exe LUDW_EFT_new.o $(LIBS)

KOBER_EFT_nn: KOBER_EFT_nn.o par/KOBER_EFT
	$(F77) $(FFLAGS) -o KOBER_EFT_nn.exe KOBER_EFT_nn.o $(LIBS)

DR2END_AK_I_2: DR2END_AK_I_2.o par/DR2END_AK_I_2
	$(F77) $(FFLAGS) -o DR2END_DD.exe DR2END_AK_I_2.o $(LIBS)

DRLUD_EFT: DRLUD_EFT.o par/DRLUD
	$(F77) $(FFLAGS) -o DRLUD_EFT.exe DRLUD_EFT.o

DROBER_EFT: DROBER_EFT.o par/DROBER
	$(F77) $(FFLAGS) -o DROBER_EFT.exe DROBER_EFT.o $(LIBS)

DRQUA_EFT: DRQUA_EFT.o par/DRQUA_AK
	$(F77) $(FFLAGS) -o DRQUA_EFT.exe DRQUA_EFT.o $(LIBS)

S-POLE_PdP: S-POLE_PdP.o par/verma coul.o
	$(F77) $(FFLAGS) -o S-POLE_PdP.exe S-POLE_PdP.o coul.o $(LIBS)
