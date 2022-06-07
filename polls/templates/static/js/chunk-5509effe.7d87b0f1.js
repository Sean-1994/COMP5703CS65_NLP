(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-5509effe"],{5933:function(t,s,e){"use strict";e("f733")},dfbd:function(t,s,e){"use strict";e.r(s);var n=function(){var t=this,s=t.$createElement,e=t._self._c||s;return e("div",[t.projectsLoading?e("div",{staticClass:"panel"},[e("div",{directives:[{name:"loading",rawName:"v-loading",value:t.projectsLoading,expression:"projectsLoading"}],staticClass:"panel-body"},[e("el-table",{attrs:{"empty-text":""}})],1)]):t._e(),t._l(t.projects,(function(s){return e("div",{key:s,staticClass:"panel"},[e("panel-title",{attrs:{title:s}}),e("div",{directives:[{name:"loading",rawName:"v-loading",value:t.projectsLoading,expression:"projectsLoading"}],staticClass:"panel-body"},[e("el-table",{directives:[{name:"loading",rawName:"v-loading",value:t.spidersLoading[s],expression:"spidersLoading[project]"}],style:{width:"100%"},attrs:{data:t.spiders[s]}},[e("el-table-column",{attrs:{prop:"id",label:t.$lang.columns.id,width:"200"}}),e("el-table-column",{attrs:{prop:"name",label:t.$lang.columns.name,width:"400"}}),e("el-table-column",{attrs:{label:t.$lang.columns.operations},scopedSlots:t._u([{key:"default",fn:function(n){return[e("el-button",{attrs:{type:"success",size:"mini"},on:{click:function(e){return t.onStartSpider(s,n.row.name)}}},[e("i",{staticClass:"fa fa-caret-right"}),t._v("\n              "+t._s(t.$lang.buttons.run)+"\n            ")])]}}],null,!0)})],1),e("el-collapse",{attrs:{accordion:""},on:{change:t.getLog}},t._l(t.jobs[s],(function(s){return e("el-collapse-item",{key:s.id,attrs:{name:s.id}},[e("template",{slot:"title"},[s.spider?e("span",{staticClass:"m-l-xs",style:{minWidth:"120px"}},[e("i",{staticClass:"fa fa-bug"}),t._v("\n              "+t._s(t.$lang.columns.spiderName)+":\n              "+t._s(s.spider)+"\n            ")]):t._e(),s.spider?e("span",{staticClass:"m-l-md",style:{minWidth:"290px"}},[e("i",{staticClass:"fa fa-key"}),t._v("\n              "+t._s(t.$lang.columns.jobID)+":\n              "+t._s(s.id)+"\n            ")]):t._e(),s.start_time?e("span",{staticClass:"m-l-md",style:{minWidth:"190px"}},[e("i",{staticClass:"el-icon-time"}),t._v("\n              "+t._s(t.$lang.columns.startTime)+":\n              "+t._s(s.start_time.substring(0,16))+"\n            ")]):t._e(),s.end_time?e("span",{staticClass:"m-l-md",style:{minWidth:"190px"}},[e("i",{staticClass:"el-icon-time"}),t._v("\n              "+t._s(t.$lang.columns.endTime)+":\n              "+t._s(s.end_time.substring(0,16))+"\n            ")]):t._e(),e("span",{staticClass:"m-l-md"},[e("el-button",{staticClass:"pull-right m-r-md",attrs:{type:t.jobStatusClass[s.status],size:"mini"}},[["pending"].includes(s.status)?e("i",{staticClass:"fa fa-circle-thin"}):t._e(),["running"].includes(s.status)?e("i",{staticClass:"fa fa-spin fa-spinner"}):t._e(),["finished"].includes(s.status)?e("i",{staticClass:"fa fa-check"}):t._e(),t._v("\n                "+t._s(t.jobStatusText[s.status])+"\n              ")]),["pending","running"].includes(s.status)?e("el-button",{staticClass:"pull-right m-r-md",attrs:{type:"danger",size:"mini"},on:{click:function(e){return e.stopPropagation(),t.onCancelJob(s.id)}}},[e("i",{staticClass:"fa fa-remove"}),["pending"].includes(s.status)?e("span",[t._v("\n                  "+t._s(t.$lang.buttons.cancel)+"\n                ")]):t._e(),["running"].includes(s.status)?e("span",[t._v("\n                  "+t._s(t.$lang.buttons.stop)+"\n                ")]):t._e()]):t._e()],1)]),e("div",{directives:[{name:"loading",rawName:"v-loading",value:t.logLoading,expression:"logLoading"}],attrs:{"element-loading-text":t.logLoadingText}},[e("pre",[t._v(t._s(t.logs[s.id]))])])],2)})),1)],1)],1)}))],2)},i=[],a=(e("ac6a"),e("eee4")),o={data:function(){return{errorCount:0,projects:[],projectsLoading:!0,spiders:{},spidersLoading:{},jobs:{},jobStatuses:["finished","running","pending"],jobStatusClass:{finished:"info",running:"success",pending:"warning"},jobStatusText:{finished:this.$store.getters.$lang.buttons.finished,running:this.$store.getters.$lang.buttons.running,pending:this.$store.getters.$lang.buttons.pending},jobsInfo:{},logs:{},logLoading:!0,logLoadingText:this.$store.getters.$lang.messages.loading,logLoadingInterval:null,logLoadingActive:null,routeId:this.$route.params.id}},components:{PanelTitle:a["a"]},created:function(){var t=this;this.getProjects(),this.$store.commit("addInterval",setInterval((function(){t.getJobs()}),5e3))},methods:{getProjects:function(){var t=this;this.projectsLoading=!0,this.$http.get(this.formatString(this.$store.state.url.client.projects,{id:this.routeId})).then((function(s){var e=s.data;t.projects=e,t.projects&&0===t.projects.length&&t.$message.info(t.$store.getters.$lang.messages.noProjects),t.projectsLoading=!1,t.projects.forEach((function(s){t.$set(t.spidersLoading,s,!0)})),t.errorCount=0,t.getSpiders(),t.getJobs()})).catch((function(){t.projectsLoading=!1,t.errorCount+=1,t.errorCount>=3?t.$message.error(t.$store.getters.$lang.messages.errorLoad):t.$store.commit("setTimeout",setTimeout((function(){t.getProjects()}),500))}))},getSpiders:function(){var t=this;this.projects.forEach((function(s){t.$http.get(t.formatString(t.$store.state.url.client.listSpiders,{id:t.routeId,project:s})).then((function(e){var n=e.data;t.$set(t.spiders,s,n),t.$set(t.spidersLoading,s,!1)})).catch((function(){t.$set(t.spidersLoading,s,!1)}))}))},getJobs:function(){var t=this;this.projects.forEach((function(s){t.$http.get(t.formatString(t.$store.state.url.client.listJobs,{id:t.routeId,project:s})).then((function(e){var n=e.data;t.$set(t.jobs,s,n);var i=function(s){var e=t.jobs[s];e.forEach((function(e){t.$set(t.jobsInfo,e.id,{project:s,spider:e["spider"]})}))};for(var a in t.jobs)i(a)}))}))},onStartSpider:function(t,s){var e=this;this.$http.get(this.formatString(this.$store.state.url.client.startSpider,{id:this.routeId,project:t,spider:s})).then((function(){e.$message.success(e.$store.getters.$lang.messages.successRun),e.getJobs()})).catch((function(){e.$message.error(e.$store.getters.$lang.messages.errorRun)}))},getLog:function(t){var s=this;t?(this.logLoadingActive=t,this.logLoading=!0,this.$http.get(this.formatString(this.$store.state.url.client.getLog,{id:this.routeId,project:this.jobsInfo[this.logLoadingActive]["project"],spider:this.jobsInfo[this.logLoadingActive]["spider"],job:this.logLoadingActive,random:Math.random()})).then((function(e){var n=e.data;s.$set(s.logs,s.logLoadingActive,n),s.logLoading=!1,s.$store.commit("setTimeout",setTimeout((function(){s.getLog(t)}),2e3))})).catch((function(){s.logLoading=!1}))):this.$store.commit("clearTimeout")},onCancelJob:function(t){var s=this;this.$http.get(this.formatString(this.$store.state.url.client.cancelJob,{id:this.routeId,project:this.jobsInfo[t]["project"],job:t})).then((function(){s.$message.success(s.$store.getters.$lang.messages.canceling),s.getJobs()})).catch((function(){s.$message.error(s.$store.getters.$lang.messages.errorCancel)}))}}},r=o,l=(e("5933"),e("2877")),c=Object(l["a"])(r,n,i,!1,null,null,null);s["default"]=c.exports},eee4:function(t,s,e){"use strict";var n=function(){var t=this,s=t.$createElement,e=t._self._c||s;return e("div",{staticClass:"panel-title"},[t.title?e("span",{domProps:{textContent:t._s(t.title)}}):t._e(),e("div",{staticClass:"fr"},[t._t("default")],2)])},i=[],a={name:"PanelTitle",props:{title:{type:String}}},o=a,r=e("2877"),l=Object(r["a"])(o,n,i,!1,null,null,null);s["a"]=l.exports},f733:function(t,s,e){}}]);
//# sourceMappingURL=chunk-5509effe.7d87b0f1.js.map