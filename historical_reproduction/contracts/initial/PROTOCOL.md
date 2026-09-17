# Five-summary prediction protocol

## 2. 冻结数学协议

### 2.1 观测

缓存L_mik=选项loglik，h_ik=字符长度，g_i=gold，选项真实数量由char_lens中有效非填充项确定。仅真实选项完整有限、h>0、gold合法的题进入；无效项不得补零。
$$u_{mik}=L_{mik}/h_{ik},\quad p_{mik}=\frac{e^{u_{mik}}}{\sum_{j\in K_i}e^{u_{mij}}},\quad \hat k_{mi}=\arg\max_k u_{mik},$$
$$y_{mi}=\mathbf1\{\hat k_{mi}=g_i\},\quad c_{mi}=p_{mi\hat k_{mi}},\quad \ell_{mi}=u_{mig_i}-\operatorname{LSE}_{k\ne g_i}(u_{mik}).$$


模型为两基准共有repo；机构由npz model_id或原repo恢复，禁止从下划线截错。无法可靠恢复机构者排除并报。源F/C各≥100有效题，目标≥200题；≥300模型、≥30机构且每外训练集≥20机构。不足→INCONCLUSIVE/INSUFFICIENT_SAMPLE，不降门槛。只按有效性及源拟合过滤，不按目标风险过滤。

### 2.2 固定分折与参数

seed=20260914；以SHA256(f"20260914|{id}")十六进制排序后按rank mod K轮转，机构外K=5，内K=4。源ARC题全局按此法K=2分F/C，不对模型各自重排。每个reference集合R与待评分H机构不交。

在R模型×源F题的有效单元ALS拟合：
$$\min_{a,s,b}\sum_{m\in R,i\in F}M_{mi}(\ell_{mi}-a_m+s_mb_i)^2,\quad \bar b=0,\ Var(b)=1.$$
题至少20个R模型有效；初始化std(−题均值ell)，每步OLS并重新定gauge；朝初始化方向定符号；max|Δb|<1e−6，上限500步。非收敛报失败、不隐式丢弃。直接用每模型有效F子集均值/方差计算：
$$s_m=-Cov_{F_m}(\ell,b)/Var_{F_m}(b),\quad a_m=\bar\ell_{F_m}+s_m\bar b_{F_m},\quad\theta_m=a_m/s_m,$$
$$\sigma_m=\sqrt{|F_m|^{-1}\sum_{i\in F_m}(\ell_{mi}-a_m+s_mb_i)^2}.$$
要求s>1e−8，sigma>1e−8，Var(ell)>1e−6且可用F≥100；所有无效源参数需列原因，两预测器使用同一合格评价集，不截尾。基本数值检查通过正常方程/独立OLS；为尊重精简不做全量双初值敏感性。

**有限嵌套规则**：外训练集A的训练特征用A内4折reference交叉拟合，外验证H用全部A作reference。内调参划A=U∪V后，U训练特征再在U内4折交叉拟合，V特征用全部U为reference；只对训练特征生成做这一次辅助4折，不在其中再调参/递归。每reference结果以机构集合SHA缓存；该缓存不得被更小训练集合违规复用。最后用A交叉拟合特征拟合预测器、H特征给外折预测。

### 2.3 两个终点与基础特征

对任意评价题集合J，N=|J|，固定箱B_j=[(j−1)/10,j/10)，j=10含1；空箱贡献0：
$$Y^{cal}=REL_{10}=\sum_{j=1}^{10}\frac{n_j}{N}(\bar c_j-\bar y_j)^2.$$
这只是有限样本top-label固定分箱可靠性操作量，不宣称连续预测Brier的精确分解。保存分箱n/cbar/ybar作为可审计组成，不增设其他主指标。

按c降序、题ID升序打破并列，取k=ceil(N/2)题组成S：
$$Y^{sel}=R_{50}=k^{-1}\sum_{i\in S}(1-y_i).$$
主Y来自目标HS全有效题；模型级参数只预测这两个标量，不给逐题排序，不声称已改进拒答算法。高置信错例不自动等于概率高估。

源ARC的全部有效题F∪C上计算
$$A=\bar y,\ C=\bar c,\ O=C-A,\quad X_B=(A,C,O,REL_{10},R_{50}),$$
$$X_F=(X_B,\theta,\log s,\log\sigma).$$
O与A/C冗余明确，ridge处理，不解读系数。基线使用全部源题，full额外参数只在源F估计，故两模型可用源题预算一致；不再声称基线统计与参数拟合题独立，外部效度来自HS目标隔离。源F/C划分仅固定参数估计子集及资格检查，不做同域泛化结论。只比较这两个模型；无自动挑最强基线、无参数消融、无T。因基线不含theta，本实验仅支持参数组联合增量，不单称s/sigma的独立贡献。比较限定线性ridge，增量不证明信息论独立，也不排除基线统计的非线性重表达能获得相同收益；本轮不加二次敏感性。

### 2.4 回归与信息权限

两个模型均为线性ridge，分别对cal/sel调lambda：
$$\hat\beta=\arg\min_{\beta_0,\beta}\sum_m v_m(Y_m-\beta_0-Z_m^T\beta)^2+\lambda\|\beta\|^2,\quad v_m=1/(G n_{g(m)}),$$
$$\lambda\in\{10^{-4},10^{-3},\ldots,10^4\},\quad\hat Y=clip(\hat\beta_0+Z^T\hat\beta,0,1).$$
训练机构等权均值/SD标准化所有连续列；训练SD≤1e−12置0；截距不惩罚；不插补无效theta/s/sigma。拼接内4折验证预测后，对全部内验证机构等权计算MSE选lambda（不是先对大小略异的四折等权取均值），差≤1e−12选较大lambda；只能用内训练数据拟合变换。内层比较lambda的模型样本固定，外比较base/full同样本。



### 2.5 推断与不可改写的终态

$$L_e(f)=G^{-1}\sum_g n_g^{-1}\sum_{m\in g}(Y_{me}-\hat Y_{me,f})^2,\quad\Delta_e=L_e(B)-L_e(F),\quad I_e=\Delta_e/L_e(B).$$
对完整OOF误差按机构配对重采样2000次(seed20260914)，保留固定预测/Y，不重训；同一抽样用于两模型/两终点。每终点97.5%双侧percentile CI=[q_.0125,q_.9875]，两主终点保守Bonferroni。条件区间只描述已训练模型/现有目标题集，不保证新题库或全训练过程覆盖；不做题bootstrap。

每终点支持条件：I点估计≥.05、Δ的97.5%配对bootstrap CI下界>0、至少4/5外折Δ>0；I的区间同时报告但不替代Δ主检验。5%为本方案操作门槛，非公认实用性标准，CI不是证明增量≥5%。
- JOINT_SUPPORT：两终点均满足；CAL_ONLY/SELECT_ONLY：仅一个满足；NO_DEMONSTRATED_GAIN：均未满足但可评价。
- INCONCLUSIVE优先于以上：样本不足、任一基线损失≤1e−12、不可计算CI；另一可评价终点照报，不用epsilon除法制造收益。
负结果不是失败，不换模型/目标/阈值重试。全数据只跑本方向，不把source/target颠倒寻找成功。

