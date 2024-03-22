---
layout: default
title: Categories
permalink: /categories/
exclude: false
---
<h2>List of categories in this blog</h2>
<ul>
{% assign categories = site.categories | sort: "title" %}
{% for post in categories %}
    <li> 
		<a class="category-name" href="{{ post.url }}">{{ post.title }}</a>
		<!--<a class="category-name" href="{{ post.url }}">
			<span style="background-color:#12486B; border:2px solid #12486B; border-radius: 5px; color:#F5FCCD">
				{{ post.title }}
		</span></a>-->
    </li>
{% endfor %}
</ul>