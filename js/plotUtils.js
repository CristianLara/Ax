function rgb(t){return"rgb("+t.join()+")"}function copy_and_reverse(t){const a=t.slice();return a.reverse(),a}function axis_range(t,a){return a?[Math.log10(Math.min(...t)),Math.log10(Math.max(...t))]:[Math.min(...t),Math.max(...t)]}function relativize_data(t,a,e,n,r){const s=!0===e?[]:t,i=!0===e?[]:a;if(!0===e){const e=n.in_sample[n.status_quo_name].y[r],o=n.in_sample[n.status_quo_name].se[r];for(let n=0;n<t.length;n++)res=relativize(t[n],a[n],e,o),s.push(100*res[0]),i.push(100*res[1])}return[s,i]}function relativize(t,a,e,n){return r_hat=(t-e)/Math.abs(e)-Math.pow(n,2)*t/Math.pow(Math.abs(e),3),variance=(Math.pow(a,2)+Math.pow(t/e*n,2))/Math.pow(e,2),[r_hat,Math.sqrt(variance)]}