%%%doing Linear Regression using BIDMach

val a = loadFMat("/Users/Anna/workspace/BIDMach_1.0.0-full-linux-x86_64/data/uci/arabic.fmat.lz4")
val c = loadFMat("/Users/Anna/workspace/BIDMach_1.0.0-full-linux-x86_64/data/uci/arabic_cats.fmat.lz4")
val (mm,mopts)=GLM.learner(a,c,1)

mopts.autoReset=false
mopts.useGPU=false

val atrain =a(?,inds(5000->a.ncols))
val atest =a(?,inds(0->5000))

val ctrain =c(?,inds(5000->a.ncols))
 val ctest =c(?,inds(0->5000))

 val cx=zeros(ctest.nrows,ctest.ncols)

val (mm,mopts,nn,nopts)=GLM.learner(atrain,ctrain,atest,cx,1)
mm.train
nn.predict

val p=ctest *@cx +(1-ctest) *@(1-cx)
mean(p,2)

val (nn,nopts)=GLM.predictor(model,atest,cx,1)
nn.predict


