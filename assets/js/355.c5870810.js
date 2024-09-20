(self.webpackChunk=self.webpackChunk||[]).push([[355],{2355:(e,t,n)=>{"use strict";n.d(t,{A:()=>Y});var o=n(8168),r=n(8587),a=n(6540),c=n(2303),s=n(53),l=n(5293),i=n(6342);function u(){var e=(0,i.p)().prism,t=(0,l.G)().colorMode,n=e.theme,o=e.darkTheme||n;return"dark"===t?o:n}var d=n(7559),p=n(8634),m=n(8426),f=n.n(m),g=(0,p.A)(/title=(["'])(.*?)\1/,{quote:1,title:2}),v=(0,p.A)(/\{([\d,-]+)\}/,{range:1}),y={js:{start:"\\/\\/",end:""},jsBlock:{start:"\\/\\*",end:"\\*\\/"},jsx:{start:"\\{\\s*\\/\\*",end:"\\*\\/\\s*\\}"},bash:{start:"#",end:""},html:{start:"\x3c!--",end:"--\x3e"}};function h(e,t){var n=e.map((function(e){var n=y[e],o=n.start,r=n.end;return"(?:"+o+"\\s*("+t.flatMap((function(e){var t,n;return[e.line,null==(t=e.block)?void 0:t.start,null==(n=e.block)?void 0:n.end].filter(Boolean)})).join("|")+")\\s*"+r+")"})).join("|");return new RegExp("^\\s*(?:"+n+")\\s*$")}function b(e,t){var n=e.replace(/\n$/,""),o=t.language,r=t.magicComments,a=t.metastring;if(a&&v.test(a)){var c=a.match(v).groups.range;if(0===r.length)throw new Error("A highlight range has been given in code block's metastring (``` "+a+"), but no magic comment config is available. Docusaurus applies the first magic comment entry's className for metastring ranges.");var s=r[0].className,l=f()(c).filter((function(e){return e>0})).map((function(e){return[e-1,[s]]}));return{lineClassNames:Object.fromEntries(l),code:n}}if(void 0===o)return{lineClassNames:{},code:n};for(var i=function(e,t){switch(e){case"js":case"javascript":case"ts":case"typescript":return h(["js","jsBlock"],t);case"jsx":case"tsx":return h(["js","jsBlock","jsx"],t);case"html":return h(["js","jsBlock","html"],t);case"python":case"py":case"bash":return h(["bash"],t);case"markdown":case"md":return h(["html","jsx","bash"],t);default:return h(Object.keys(y),t)}}(o,r),u=n.split("\n"),d=Object.fromEntries(r.map((function(e){return[e.className,{start:0,range:""}]}))),p=Object.fromEntries(r.filter((function(e){return e.line})).map((function(e){var t=e.className;return[e.line,t]}))),m=Object.fromEntries(r.filter((function(e){return e.block})).map((function(e){var t=e.className;return[e.block.start,t]}))),g=Object.fromEntries(r.filter((function(e){return e.block})).map((function(e){var t=e.className;return[e.block.end,t]}))),b=0;b<u.length;){var k=u[b].match(i);if(k){var E=k.slice(1).find((function(e){return void 0!==e}));p[E]?d[p[E]].range+=b+",":m[E]?d[m[E]].start=b:g[E]&&(d[g[E]].range+=d[g[E]].start+"-"+(b-1)+","),u.splice(b,1)}else b+=1}n=u.join("\n");var B={};return Object.entries(d).forEach((function(e){var t=e[0],n=e[1].range;f()(n).forEach((function(e){null!=B[e]||(B[e]=[]),B[e].push(t)}))})),{lineClassNames:B,code:n}}const k={codeBlockContainer:"codeBlockContainer_Ckt0"};var E=["as"];function B(e){var t=e.as,n=(0,r.A)(e,E),c=function(e){var t={color:"--prism-color",backgroundColor:"--prism-background-color"},n={};return Object.entries(e.plain).forEach((function(e){var o=e[0],r=e[1],a=t[o];a&&"string"==typeof r&&(n[a]=r)})),n}(u());return a.createElement(t,(0,o.A)({},n,{style:c,className:(0,s.A)(n.className,k.codeBlockContainer,d.G.common.codeBlock)}))}const N={codeBlockContent:"codeBlockContent_biex",codeBlockTitle:"codeBlockTitle_Ktv7",codeBlock:"codeBlock_bY9V",codeBlockStandalone:"codeBlockStandalone_MEMb",codeBlockLines:"codeBlockLines_e6Vv",codeBlockLinesWithNumbering:"codeBlockLinesWithNumbering_o6Pm",buttonGroup:"buttonGroup__atx"};function C(e){var t=e.children,n=e.className;return a.createElement(B,{as:"pre",tabIndex:0,className:(0,s.A)(N.codeBlockStandalone,"thin-scrollbar",n)},a.createElement("code",{className:N.codeBlockLines},t))}var w=n(3807),L={attributes:!0,characterData:!0,childList:!0,subtree:!0};function A(e,t){var n=(0,a.useState)(),o=n[0],r=n[1],c=(0,a.useCallback)((function(){var t;r(null==(t=e.current)?void 0:t.closest("[role=tabpanel][hidden]"))}),[e,r]);(0,a.useEffect)((function(){c()}),[c]),function(e,t,n){void 0===n&&(n=L);var o=(0,w._q)(t),r=(0,w.Be)(n);(0,a.useEffect)((function(){var t=new MutationObserver(o);return e&&t.observe(e,r),function(){return t.disconnect()}}),[e,o,r])}(o,(function(e){e.forEach((function(e){"attributes"===e.type&&"hidden"===e.attributeName&&(t(),c())}))}),{attributes:!0,characterData:!1,childList:!1,subtree:!1})}const j={plain:{backgroundColor:"#2a2734",color:"#9a86fd"},styles:[{types:["comment","prolog","doctype","cdata","punctuation"],style:{color:"#6c6783"}},{types:["namespace"],style:{opacity:.7}},{types:["tag","operator","number"],style:{color:"#e09142"}},{types:["property","function"],style:{color:"#9a86fd"}},{types:["tag-id","selector","atrule-id"],style:{color:"#eeebff"}},{types:["attr-name"],style:{color:"#c4b9fe"}},{types:["boolean","string","entity","url","attr-value","keyword","control","directive","unit","statement","regex","atrule","placeholder","variable"],style:{color:"#ffcc99"}},{types:["deleted"],style:{textDecorationLine:"line-through"}},{types:["inserted"],style:{textDecorationLine:"underline"}},{types:["italic"],style:{fontStyle:"italic"}},{types:["important","bold"],style:{fontWeight:"bold"}},{types:["important"],style:{color:"#c4b9fe"}}]};var T={Prism:n(1258).A,theme:j};function x(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function S(){return S=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var o in n)Object.prototype.hasOwnProperty.call(n,o)&&(e[o]=n[o])}return e},S.apply(this,arguments)}var _=/\r\n|\r|\n/,O=function(e){0===e.length?e.push({types:["plain"],content:"\n",empty:!0}):1===e.length&&""===e[0].content&&(e[0].content="\n",e[0].empty=!0)},P=function(e,t){var n=e.length;return n>0&&e[n-1]===t?e:e.concat(t)};function I(e,t){var n={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&-1===t.indexOf(o)&&(n[o]=e[o]);return n}var W=function(e){function t(){for(var t=this,n=[],o=arguments.length;o--;)n[o]=arguments[o];e.apply(this,n),x(this,"getThemeDict",(function(e){if(void 0!==t.themeDict&&e.theme===t.prevTheme&&e.language===t.prevLanguage)return t.themeDict;t.prevTheme=e.theme,t.prevLanguage=e.language;var n=e.theme?function(e,t){var n=e.plain,o=Object.create(null),r=e.styles.reduce((function(e,n){var o=n.languages,r=n.style;return o&&!o.includes(t)||n.types.forEach((function(t){var n=S({},e[t],r);e[t]=n})),e}),o);return r.root=n,r.plain=S({},n,{backgroundColor:null}),r}(e.theme,e.language):void 0;return t.themeDict=n})),x(this,"getLineProps",(function(e){var n=e.key,o=e.className,r=e.style,a=S({},I(e,["key","className","style","line"]),{className:"token-line",style:void 0,key:void 0}),c=t.getThemeDict(t.props);return void 0!==c&&(a.style=c.plain),void 0!==r&&(a.style=void 0!==a.style?S({},a.style,r):r),void 0!==n&&(a.key=n),o&&(a.className+=" "+o),a})),x(this,"getStyleForToken",(function(e){var n=e.types,o=e.empty,r=n.length,a=t.getThemeDict(t.props);if(void 0!==a){if(1===r&&"plain"===n[0])return o?{display:"inline-block"}:void 0;if(1===r&&!o)return a[n[0]];var c=o?{display:"inline-block"}:{},s=n.map((function(e){return a[e]}));return Object.assign.apply(Object,[c].concat(s))}})),x(this,"getTokenProps",(function(e){var n=e.key,o=e.className,r=e.style,a=e.token,c=S({},I(e,["key","className","style","token"]),{className:"token "+a.types.join(" "),children:a.content,style:t.getStyleForToken(a),key:void 0});return void 0!==r&&(c.style=void 0!==c.style?S({},c.style,r):r),void 0!==n&&(c.key=n),o&&(c.className+=" "+o),c})),x(this,"tokenize",(function(e,t,n,o){var r={code:t,grammar:n,language:o,tokens:[]};e.hooks.run("before-tokenize",r);var a=r.tokens=e.tokenize(r.code,r.grammar,r.language);return e.hooks.run("after-tokenize",r),a}))}return e&&(t.__proto__=e),t.prototype=Object.create(e&&e.prototype),t.prototype.constructor=t,t.prototype.render=function(){var e=this.props,t=e.Prism,n=e.language,o=e.code,r=e.children,a=this.getThemeDict(this.props),c=t.languages[n];return r({tokens:function(e){for(var t=[[]],n=[e],o=[0],r=[e.length],a=0,c=0,s=[],l=[s];c>-1;){for(;(a=o[c]++)<r[c];){var i=void 0,u=t[c],d=n[c][a];if("string"==typeof d?(u=c>0?u:["plain"],i=d):(u=P(u,d.type),d.alias&&(u=P(u,d.alias)),i=d.content),"string"==typeof i){var p=i.split(_),m=p.length;s.push({types:u,content:p[0]});for(var f=1;f<m;f++)O(s),l.push(s=[]),s.push({types:u,content:p[f]})}else c++,t.push(u),n.push(i),o.push(0),r.push(i.length)}c--,t.pop(),n.pop(),o.pop(),r.pop()}return O(s),l}(void 0!==c?this.tokenize(t,o,c,n):[o]),className:"prism-code language-"+n,style:void 0!==a?a.root:{},getLineProps:this.getLineProps,getTokenProps:this.getTokenProps})},t}(a.Component);const D=W,H={codeLine:"codeLine_lJS_",codeLineNumber:"codeLineNumber_Tfdd",codeLineContent:"codeLineContent_feaV"};function z(e){var t=e.line,n=e.classNames,r=e.showLineNumbers,c=e.getLineProps,l=e.getTokenProps;1===t.length&&"\n"===t[0].content&&(t[0].content="");var i=c({line:t,className:(0,s.A)(n,r&&H.codeLine)}),u=t.map((function(e,t){return a.createElement("span",(0,o.A)({key:t},l({token:e,key:t})))}));return a.createElement("span",i,r?a.createElement(a.Fragment,null,a.createElement("span",{className:H.codeLineNumber}),a.createElement("span",{className:H.codeLineContent},u)):u,a.createElement("br",null))}var M=n(1312);function V(e){return a.createElement("svg",(0,o.A)({viewBox:"0 0 24 24"},e),a.createElement("path",{fill:"currentColor",d:"M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"}))}function R(e){return a.createElement("svg",(0,o.A)({viewBox:"0 0 24 24"},e),a.createElement("path",{fill:"currentColor",d:"M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z"}))}const G={copyButtonCopied:"copyButtonCopied_obH4",copyButtonIcons:"copyButtonIcons_eSgA",copyButtonIcon:"copyButtonIcon_y97N",copyButtonSuccessIcon:"copyButtonSuccessIcon_LjdS"};function q(e){var t=e.code,n=e.className,o=(0,a.useState)(!1),r=o[0],c=o[1],l=(0,a.useRef)(void 0),i=(0,a.useCallback)((function(){!function(e,t){var n=(void 0===t?{}:t).target,o=void 0===n?document.body:n;if("string"!=typeof e)throw new TypeError("Expected parameter `text` to be a `string`, got `"+typeof e+"`.");var r=document.createElement("textarea"),a=document.activeElement;r.value=e,r.setAttribute("readonly",""),r.style.contain="strict",r.style.position="absolute",r.style.left="-9999px",r.style.fontSize="12pt";var c=document.getSelection(),s=c.rangeCount>0&&c.getRangeAt(0);o.append(r),r.select(),r.selectionStart=0,r.selectionEnd=e.length;var l=!1;try{l=document.execCommand("copy")}catch(i){}r.remove(),s&&(c.removeAllRanges(),c.addRange(s)),a&&a.focus()}(t),c(!0),l.current=window.setTimeout((function(){c(!1)}),1e3)}),[t]);return(0,a.useEffect)((function(){return function(){return window.clearTimeout(l.current)}}),[]),a.createElement("button",{type:"button","aria-label":r?(0,M.T)({id:"theme.CodeBlock.copied",message:"Copied",description:"The copied button label on code blocks"}):(0,M.T)({id:"theme.CodeBlock.copyButtonAriaLabel",message:"Copy code to clipboard",description:"The ARIA label for copy code blocks button"}),title:(0,M.T)({id:"theme.CodeBlock.copy",message:"Copy",description:"The copy button label on code blocks"}),className:(0,s.A)("clean-btn",n,G.copyButton,r&&G.copyButtonCopied),onClick:i},a.createElement("span",{className:G.copyButtonIcons,"aria-hidden":"true"},a.createElement(V,{className:G.copyButtonIcon}),a.createElement(R,{className:G.copyButtonSuccessIcon})))}function F(e){return a.createElement("svg",(0,o.A)({viewBox:"0 0 24 24"},e),a.createElement("path",{fill:"currentColor",d:"M4 19h6v-2H4v2zM20 5H4v2h16V5zm-3 6H4v2h13.25c1.1 0 2 .9 2 2s-.9 2-2 2H15v-2l-3 3l3 3v-2h2c2.21 0 4-1.79 4-4s-1.79-4-4-4z"}))}const $={wordWrapButtonIcon:"wordWrapButtonIcon_Bwma",wordWrapButtonEnabled:"wordWrapButtonEnabled_EoeP"};function Z(e){var t=e.className,n=e.onClick,o=e.isEnabled,r=(0,M.T)({id:"theme.CodeBlock.wordWrapToggle",message:"Toggle word wrap",description:"The title attribute for toggle word wrapping button of code block lines"});return a.createElement("button",{type:"button",onClick:n,className:(0,s.A)("clean-btn",t,o&&$.wordWrapButtonEnabled),"aria-label":r,title:r},a.createElement(F,{className:$.wordWrapButtonIcon,"aria-hidden":"true"}))}function J(e){var t,n,r,c,l,d,p,m,f,v,y,h=e.children,k=e.className,E=void 0===k?"":k,C=e.metastring,w=e.title,L=e.showLineNumbers,j=e.language,x=(0,i.p)().prism,S=x.defaultLanguage,_=x.magicComments,O=null!=(t=null!=j?j:null==(n=E.split(" ").find((function(e){return e.startsWith("language-")})))?void 0:n.replace(/language-/,""))?t:S,P=u(),I=(r=(0,a.useState)(!1),c=r[0],l=r[1],d=(0,a.useState)(!1),p=d[0],m=d[1],f=(0,a.useRef)(null),v=(0,a.useCallback)((function(){var e=f.current.querySelector("code");c?e.removeAttribute("style"):(e.style.whiteSpace="pre-wrap",e.style.overflowWrap="anywhere"),l((function(e){return!e}))}),[f,c]),y=(0,a.useCallback)((function(){var e=f.current,t=e.scrollWidth>e.clientWidth||f.current.querySelector("code").hasAttribute("style");m(t)}),[f]),A(f,y),(0,a.useEffect)((function(){y()}),[c,y]),(0,a.useEffect)((function(){return window.addEventListener("resize",y,{passive:!0}),function(){window.removeEventListener("resize",y)}}),[y]),{codeBlockRef:f,isEnabled:c,isCodeScrollable:p,toggle:v}),W=function(e){var t,n;return null!=(t=null==e||null==(n=e.match(g))?void 0:n.groups.title)?t:""}(C)||w,H=b(h,{metastring:C,language:O,magicComments:_}),M=H.lineClassNames,V=H.code,R=null!=L?L:function(e){return Boolean(null==e?void 0:e.includes("showLineNumbers"))}(C);return a.createElement(B,{as:"div",className:(0,s.A)(E,O&&!E.includes("language-"+O)&&"language-"+O)},W&&a.createElement("div",{className:N.codeBlockTitle},W),a.createElement("div",{className:N.codeBlockContent},a.createElement(D,(0,o.A)({},T,{theme:P,code:V,language:null!=O?O:"text"}),(function(e){var t=e.className,n=e.tokens,o=e.getLineProps,r=e.getTokenProps;return a.createElement("pre",{tabIndex:0,ref:I.codeBlockRef,className:(0,s.A)(t,N.codeBlock,"thin-scrollbar")},a.createElement("code",{className:(0,s.A)(N.codeBlockLines,R&&N.codeBlockLinesWithNumbering)},n.map((function(e,t){return a.createElement(z,{key:t,line:e,getLineProps:o,getTokenProps:r,classNames:M[t],showLineNumbers:R})}))))})),a.createElement("div",{className:N.buttonGroup},(I.isEnabled||I.isCodeScrollable)&&a.createElement(Z,{className:N.codeButton,onClick:function(){return I.toggle()},isEnabled:I.isEnabled}),a.createElement(q,{className:N.codeButton,code:V}))))}var K=["children"];function Y(e){var t=e.children,n=(0,r.A)(e,K),s=(0,c.A)(),l=function(e){return a.Children.toArray(e).some((function(e){return(0,a.isValidElement)(e)}))?e:Array.isArray(e)?e.join(""):e}(t),i="string"==typeof l?J:C;return a.createElement(i,(0,o.A)({key:String(s)},n),l)}},8426:(e,t)=>{function n(e){let t,n=[];for(let o of e.split(",").map((e=>e.trim())))if(/^-?\d+$/.test(o))n.push(parseInt(o,10));else if(t=o.match(/^(-?\d+)(-|\.\.\.?|\u2025|\u2026|\u22EF)(-?\d+)$/)){let[e,o,r,a]=t;if(o&&a){o=parseInt(o),a=parseInt(a);const e=o<a?1:-1;"-"!==r&&".."!==r&&"\u2025"!==r||(a+=e);for(let t=o;t!==a;t+=e)n.push(t)}}return n}t.default=n,e.exports=n}}]);