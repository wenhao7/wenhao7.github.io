---
layout: default
title: Tags
permalink: /tags/
exclude: false
---
<h2>List of Tags in this blog</h2>
<ul>
{% capture temptags %}
  {% for tag in site.tags %}
    {{ tag[0] }}#{{ tag[1].size }}
  {% endfor %}
{% endcapture %}
{% assign sortedtemptags = temptags | split:' ' | sort %}
{% for temptag in sortedtemptags %}
  {% assign tagitems = temptag | split: '#' %}
  <!--<a href="/tag/{{ tagname }}"><code class="highligher-rouge"><nobr>{{ tagname    }}</nobr></code></a>-->
  <a href="/tag/{{ tagitems[0] }}"><code class="highlighter-tag"><nobr>{{ tagitems[0] }} ({{tagitems[1]}})</nobr></code></a>
  <!--<a href="/tag/{{ tagname }}">
	<span style="background-color:#007F73; border:2px solid #007F73; border-radius: 5px; color:#F5FCCD">
		{{	tagname		}}
	</span></a>-->
{% endfor %}
</ul>

<!--
{% capture temptags %}
  {% for tag in site.tags %}
    {{ tag[1].size | plus: 1000 }}#{{ tag[0] }}#{{ tag[1].size }}
  {% endfor %}
{% endcapture %}
{% assign sortedtemptags = temptags | split:' ' | sort | reverse %}
{% for temptag in sortedtemptags %}
  {% assign tagitems = temptag | split: '#' %}
  {% capture tagname %}{{ tagitems[1] }}{% endcapture %}
  <a href="/tag/{{ tagname }}"><code class="highlighter-rouge"><nobr>{{ tagname    }}</nobr></code></a>

{% endfor %}
-->