---
layout: default
title: Categories
permalink: /categories/
exclude: false
---
<h2>List of Categories in this blog</h2>
<ul>
{% assign categories = site.categories | sort: "title" %}
{% for post in categories %}
    <li> 
		<a href="{{ post.url }}"><h3>{{ post.title | replace: "_", " " }}</h3></a>
		<!--<a class="category-name" href="{{ post.url }}">
			<span style="background-color:#12486B; border:2px solid #12486B; border-radius: 5px; color:#F5FCCD">
				{{ post.title }}
		</span></a>-->
    </li>
{% endfor %}
</ul>