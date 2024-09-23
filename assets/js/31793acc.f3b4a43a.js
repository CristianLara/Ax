"use strict";(self.webpackChunk=self.webpackChunk||[]).push([[505],{6313:(e,r,a)=>{a.r(r),a.d(r,{assets:()=>l,contentTitle:()=>s,default:()=>h,frontMatter:()=>n,metadata:()=>o,toc:()=>c});var i=a(4848),t=a(8453);const n={id:"glossary",title:"Glossary",sidebar_label:"Glossary"},s=void 0,o={id:"glossary",title:"Glossary",description:"Arm",source:"@site/../docs/glossary.md",sourceDirName:".",slug:"/glossary",permalink:"/Ax/docs/glossary",draft:!1,unlisted:!1,tags:[],version:"current",lastUpdatedBy:"Cristian Lara",lastUpdatedAt:1727124263e3,frontMatter:{id:"glossary",title:"Glossary",sidebar_label:"Glossary"},sidebar:"docs",previous:{title:"APIs",permalink:"/Ax/docs/api"},next:{title:"Bayesian Optimization",permalink:"/Ax/docs/bayesopt"}},l={},c=[{value:"Arm",id:"arm",level:3},{value:"Bandit optimization",id:"bandit-optimization",level:3},{value:"Batch trial",id:"batch-trial",level:3},{value:"Bayesian optimization",id:"bayesian-optimization",level:3},{value:"Evaluation function",id:"evaluation-function",level:3},{value:"Experiment",id:"experiment",level:3},{value:"Generation strategy",id:"generation-strategy",level:3},{value:"Generator run",id:"generator-run",level:3},{value:"Metric",id:"metric",level:3},{value:"Model",id:"model",level:3},{value:"Model bridge",id:"model-bridge",level:3},{value:"Objective",id:"objective",level:3},{value:"Optimization config",id:"optimization-config",level:3},{value:"Outcome constraint",id:"outcome-constraint",level:3},{value:"Parameter",id:"parameter",level:3},{value:"Parameter constraint",id:"parameter-constraint",level:3},{value:"Relative outcome constraint",id:"relative-outcome-constraint",level:3},{value:"Runner",id:"runner",level:3},{value:"Scheduler",id:"scheduler",level:3},{value:"Search space",id:"search-space",level:3},{value:"SEM",id:"sem",level:3},{value:"Status quo",id:"status-quo",level:3},{value:"Trial",id:"trial",level:3}];function d(e){const r={a:"a",code:"code",h3:"h3",p:"p",strong:"strong",...(0,t.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(r.h3,{id:"arm",children:"Arm"}),"\n",(0,i.jsxs)(r.p,{children:["Mapping from ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#parameter",children:"parameters"})," (i.e. a parameterization or parameter configuration) to parameter values. An arm provides the configuration to be tested in an Ax ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#trial",children:"trial"}),'. Also known as "treatment group" or "parameterization", the name \'arm\' comes from the ',(0,i.jsx)(r.a,{href:"https://en.wikipedia.org/wiki/Multi-armed_bandit",children:"Multi-Armed Bandit"})," optimization problem, in which a player facing a row of \u201cone-armed bandit\u201d slot machines has to choose which machines to play when and in what order. ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.arm",children:(0,i.jsx)(r.code,{children:"[Arm]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"bandit-optimization",children:"Bandit optimization"}),"\n",(0,i.jsxs)(r.p,{children:["Machine learning-driven version of A/B testing that dynamically allocates traffic to ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arms"})," which are performing well, to determine the best ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arm"})," among a given set."]}),"\n",(0,i.jsx)(r.h3,{id:"batch-trial",children:"Batch trial"}),"\n",(0,i.jsxs)(r.p,{children:["Single step in the ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#experiment",children:"experiment"}),", contains multiple ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arms"})," that are ",(0,i.jsx)(r.strong,{children:"deployed and evaluated together"}),". A batch trial is not just a trial with many arms; it is a trial for which it is important that the arms are evaluated simultaneously, e.g. in an A/B test where the evaluation results are subject to nonstationarity. For cases where multiple arms are evaluated separately and independently of each other, use multiple regular ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#trial",children:"trials"})," with a single arm each. ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.batch_trial",children:(0,i.jsx)(r.code,{children:"[BatchTrial]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"bayesian-optimization",children:"Bayesian optimization"}),"\n",(0,i.jsxs)(r.p,{children:["Sequential optimization strategy for finding an optimal ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arm"})," in a continuous ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#search-space",children:"search space"}),"."]}),"\n",(0,i.jsx)(r.h3,{id:"evaluation-function",children:"Evaluation function"}),"\n",(0,i.jsxs)(r.p,{children:["Function that takes a parameterization and an optional weight as input and outputs a set of metric evaluations (",(0,i.jsx)(r.a,{href:"/Ax/docs/trial-evaluation#evaluation-function",children:"more details"}),"). Used in the ",(0,i.jsx)(r.a,{href:"/Ax/docs/api",children:"Loop API"}),"."]}),"\n",(0,i.jsx)(r.h3,{id:"experiment",children:"Experiment"}),"\n",(0,i.jsxs)(r.p,{children:["Object that keeps track of the whole optimization process. Contains a ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#search-space",children:"search space"}),", ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#optimization-config",children:"optimization config"}),", and other metadata. ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.experiment",children:(0,i.jsx)(r.code,{children:"[Experiment]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"generation-strategy",children:"Generation strategy"}),"\n",(0,i.jsxs)(r.p,{children:["Abstraction that allows to declaratively specify one or multiple models to use in the course of the optimization and automate transition between them (relevant ",(0,i.jsx)(r.a,{href:"/tutorials/scheduler.html",children:"tutorial"}),"). ",(0,i.jsx)(r.a,{href:"/api/modelbridge.html#module-ax.modelbridge.generation_strategy",children:(0,i.jsx)(r.code,{children:"[GenerationStrategy]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"generator-run",children:"Generator run"}),"\n",(0,i.jsxs)(r.p,{children:["Outcome of a single run of the ",(0,i.jsx)(r.code,{children:"gen"})," method of a ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#model-bridge",children:"model bridge"}),", contains the generated ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arms"}),", as well as possibly best ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arm"})," predictions, other ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#model",children:"model"})," predictions, fit times etc. ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.generator_run",children:(0,i.jsx)(r.code,{children:"[GeneratorRun]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"metric",children:"Metric"}),"\n",(0,i.jsxs)(r.p,{children:["Interface for fetching data for a specific measurement on an ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#experiment",children:"experiment"})," or ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#trial",children:"trial"}),". ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.metric",children:(0,i.jsx)(r.code,{children:"[Metric]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"model",children:"Model"}),"\n",(0,i.jsxs)(r.p,{children:["Algorithm that can be used to generate new points in a ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#search-space",children:"search space"}),". ",(0,i.jsx)(r.a,{href:"/api/models.html",children:(0,i.jsx)(r.code,{children:"[Model]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"model-bridge",children:"Model bridge"}),"\n",(0,i.jsxs)(r.p,{children:["Adapter for interactions with a ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#model",children:"model"})," within the Ax ecosystem. ",(0,i.jsx)(r.a,{href:"/api/modelbridge.html",children:(0,i.jsx)(r.code,{children:"[ModelBridge]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"objective",children:"Objective"}),"\n",(0,i.jsxs)(r.p,{children:["The ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#metric",children:"metric"})," to be optimized, with an optimization direction (maximize/minimize). ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.objective",children:(0,i.jsx)(r.code,{children:"[Objective]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"optimization-config",children:"Optimization config"}),"\n",(0,i.jsxs)(r.p,{children:["Contains information necessary to run an optimization, i.e. ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#objective",children:"objective"})," and ",(0,i.jsx)(r.a,{href:"glossary#outcome-constraints",children:"outcome constraints"}),". ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.optimization_config",children:(0,i.jsx)(r.code,{children:"[OptimizationConfig]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"outcome-constraint",children:"Outcome constraint"}),"\n",(0,i.jsxs)(r.p,{children:["Constraint on ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#metric",children:"metric"})," values, can be an order constraint or a sum constraint; violating ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arms"})," will be considered infeasible. ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.outcome_constraint",children:(0,i.jsx)(r.code,{children:"[OutcomeConstraint]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"parameter",children:"Parameter"}),"\n",(0,i.jsxs)(r.p,{children:["Configurable quantity that can be assigned one of multiple possible values, can be continuous (",(0,i.jsx)(r.a,{href:"../api/core.html#ax.core.parameter.RangeParameter",children:(0,i.jsx)(r.code,{children:"RangeParameter"})}),"), discrete (",(0,i.jsx)(r.a,{href:"../api/core.html#ax.core.parameter.ChoiceParameter",children:(0,i.jsx)(r.code,{children:"ChoiceParameter"})}),") or fixed (",(0,i.jsx)(r.a,{href:"../api/core.html#ax.core.parameter.FixedParameter",children:(0,i.jsx)(r.code,{children:"FixedParameter"})}),"). ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.parameter",children:(0,i.jsx)(r.code,{children:"[Parameter]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"parameter-constraint",children:"Parameter constraint"}),"\n",(0,i.jsxs)(r.p,{children:["Places restrictions on the relationships between ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#parameter",children:"parameters"}),".  For example ",(0,i.jsx)(r.code,{children:"buffer_size1 < buffer_size2"})," or ",(0,i.jsx)(r.code,{children:"buffer_size_1 + buffer_size_2 < 1024"}),". ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.parameter_constraint",children:(0,i.jsx)(r.code,{children:"[ParameterConstraint]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"relative-outcome-constraint",children:"Relative outcome constraint"}),"\n",(0,i.jsxs)(r.p,{children:[(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#outcome-constraint",children:"Outcome constraint"})," evaluated relative to the ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#status-quo",children:"status quo"})," instead of directly on the metric value. ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.outcome_constraint",children:(0,i.jsx)(r.code,{children:"[OutcomeConstraint]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"runner",children:"Runner"}),"\n",(0,i.jsxs)(r.p,{children:["Dispatch abstraction that defines how a given ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#trial",children:"trial"})," is to be run (either locally or by dispatching to an external system). ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.runner",children:(0,i.jsx)(r.code,{children:"[Runner]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"scheduler",children:"Scheduler"}),"\n",(0,i.jsxs)(r.p,{children:["Configurable closed-loop optimization manager class, capable of conducting a full experiment by deploying trials, polling their results, and leveraging those results to generate and deploy more\ntrials (relevant ",(0,i.jsx)(r.a,{href:"/tutorials/scheduler.html",children:"tutorial"}),"). ",(0,i.jsx)(r.a,{href:"https://ax.dev/versions/latest/api/service.html#module-ax.service.scheduler",children:(0,i.jsx)(r.code,{children:"[Scheduler]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"search-space",children:"Search space"}),"\n",(0,i.jsxs)(r.p,{children:["Continuous, discrete or mixed design space that defines the set of ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#parameter",children:"parameters"})," to be tuned in the optimization, and optionally ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#parameter-constraint",children:"parameter constraints"})," on these parameters. The parameters of the ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arms"})," to be evaluated in the optimization are drawn from a search space. ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.search_space",children:(0,i.jsx)(r.code,{children:"[SearchSpace]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"sem",children:"SEM"}),"\n",(0,i.jsxs)(r.p,{children:[(0,i.jsx)(r.a,{href:"https://en.wikipedia.org/wiki/Standard_error",children:"Standard error"})," of the ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#metric",children:"metric"}),"'s mean, 0.0 for noiseless measurements. If no value is provided, defaults to ",(0,i.jsx)(r.code,{children:"np.nan"}),", in which case Ax infers its value using the measurements collected during experimentation."]}),"\n",(0,i.jsx)(r.h3,{id:"status-quo",children:"Status quo"}),"\n",(0,i.jsxs)(r.p,{children:["An ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arm"}),", usually the currently deployed configuration, which provides a baseline for comparing all other ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arms"}),". Also known as a control ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arm"}),". ",(0,i.jsx)(r.a,{href:"/api/core.html#ax.core.experiment.Experiment.status_quo",children:(0,i.jsx)(r.code,{children:"[StatusQuo]"})})]}),"\n",(0,i.jsx)(r.h3,{id:"trial",children:"Trial"}),"\n",(0,i.jsxs)(r.p,{children:["Single step in the ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#experiment",children:"experiment"}),", contains a single ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arm"}),". In cases where the trial contains multiple ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#arm",children:"arms"})," that are deployed simultaneously, we refer to it as a ",(0,i.jsx)(r.a,{href:"/Ax/docs/glossary#batch-trial",children:"batch trial"}),". ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.trial",children:(0,i.jsx)(r.code,{children:"[Trial]"})}),", ",(0,i.jsx)(r.a,{href:"/api/core.html#module-ax.core.batch_trial",children:(0,i.jsx)(r.code,{children:"[BatchTrial]"})})]})]})}function h(e={}){const{wrapper:r}={...(0,t.R)(),...e.components};return r?(0,i.jsx)(r,{...e,children:(0,i.jsx)(d,{...e})}):d(e)}},8453:(e,r,a)=>{a.d(r,{R:()=>s,x:()=>o});var i=a(6540);const t={},n=i.createContext(t);function s(e){const r=i.useContext(n);return i.useMemo((function(){return"function"==typeof e?e(r):{...r,...e}}),[r,e])}function o(e){let r;return r=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:s(e.components),i.createElement(n.Provider,{value:r},e.children)}}}]);