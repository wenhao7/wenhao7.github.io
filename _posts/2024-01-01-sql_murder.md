---
layout: post
title:  "SQL Murder Mystery"
date:   2024-01-01 22:31:10 +0800
category: [data_analysis, misc]
tag: [sql, fun]
summary: "A fun little game where we play the role of a detective querying an SQL database to solve a murder mystery."
---

In this notebook we will be solving the [SQL Murder Mystery](https://mystery.knightlab.com) designed by [Knight Lab](https://knightlab.northwestern.edu), a fun little game where we have to solve the crime by querying the database.

To begin with we will connect to the database. The %sql magic allows us to issue SQL commands within this notebook.


```python
%load_ext sql
```


```python
%sql sqlite:///sql-murder-mystery.db
```

## Prompt
A prompt is provided to get us started on solving this mystery.

>A crime has taken place and the detective needs your help. The detective gave you the 
crime scene report, but you somehow lost it. You vaguely remember that the crime 
was a **murder** that occurred sometime on **Jan.15, 2018** and that it took place in **SQL 
City**. Start by retrieving the corresponding crime scene report from the police 
department’s database.


```python
%sql SELECT name \
FROM sqlite_master \
WHERE type = 'table'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>name</th>
    </tr>
    <tr>
        <td>crime_scene_report</td>
    </tr>
    <tr>
        <td>drivers_license</td>
    </tr>
    <tr>
        <td>person</td>
    </tr>
    <tr>
        <td>facebook_event_checkin</td>
    </tr>
    <tr>
        <td>interview</td>
    </tr>
    <tr>
        <td>get_fit_now_member</td>
    </tr>
    <tr>
        <td>get_fit_now_check_in</td>
    </tr>
    <tr>
        <td>income</td>
    </tr>
    <tr>
        <td>solution</td>
    </tr>
</table>



The above shows the different tables available in the database. According to the prompt, crime_scene_report should be the first table we explore.

## Crime Scene Report


```python
%sql SELECT sql \
FROM sqlite_master \
WHERE name = 'crime_scene_report'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>sql</th>
    </tr>
    <tr>
        <td>CREATE TABLE crime_scene_report (<br>        date integer,<br>        type text,<br>        description text,<br>        city text<br>    )</td>
    </tr>
</table>



This query will tell us the column names and the corresponding data types inside the table, allowing us to infer the table and make additional queries.


```python
%sql SELECT * \
FROM crime_scene_report \
WHERE type = 'murder' \
AND date BETWEEN 20180101 AND 20180131 \
AND lower(city) = 'sql city'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>date</th>
        <th>type</th>
        <th>description</th>
        <th>city</th>
    </tr>
    <tr>
        <td>20180115</td>
        <td>murder</td>
        <td>Security footage shows that there were 2 witnesses. The first witness lives at the last house on &quot;Northwestern Dr&quot;. The second witness, named Annabel, lives somewhere on &quot;Franklin Ave&quot;.</td>
        <td>SQL City</td>
    </tr>
</table>



Inserting the conditions found in the prompt, we find only one crime scene report that matches. The description mentions two witnesses which we will now find.

## Witness 1


```python
%sql SELECT sql \
FROM sqlite_master \
WHERE name = 'person'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>sql</th>
    </tr>
    <tr>
        <td>CREATE TABLE person (<br>        id integer PRIMARY KEY,<br>        name text,<br>        license_id integer,<br>        address_number integer,<br>        address_street_name text,<br>        ssn integer,<br>        FOREIGN KEY (license_id) REFERENCES drivers_license(id)<br>    )</td>
    </tr>
</table>




```python
%sql SELECT * \
FROM person \
WHERE lower(address_street_name) = 'northwestern dr'\
ORDER BY address_number DESC \
LIMIT 1
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>id</th>
        <th>name</th>
        <th>license_id</th>
        <th>address_number</th>
        <th>address_street_name</th>
        <th>ssn</th>
    </tr>
    <tr>
        <td>14887</td>
        <td>Morty Schapiro</td>
        <td>118009</td>
        <td>4919</td>
        <td>Northwestern Dr</td>
        <td>111564949</td>
    </tr>
</table>



The first witness living in the last house on Northwestern Dr is Morty Schapiro.

## Witness 2


```python
%sql SELECT * \
FROM person \
WHERE lower(address_street_name) = 'franklin ave' \
AND lower(name) LIKE '%annabel%'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>id</th>
        <th>name</th>
        <th>license_id</th>
        <th>address_number</th>
        <th>address_street_name</th>
        <th>ssn</th>
    </tr>
    <tr>
        <td>16371</td>
        <td>Annabel Miller</td>
        <td>490173</td>
        <td>103</td>
        <td>Franklin Ave</td>
        <td>318771143</td>
    </tr>
</table>



The second witness living on Franklin Ave is Annabel Miller.

## Interview Transcripts


```python
%sql SELECT sql \
FROM sqlite_master \
WHERE name = 'interview'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>sql</th>
    </tr>
    <tr>
        <td>CREATE TABLE interview (<br>        person_id integer,<br>        transcript text,<br>        FOREIGN KEY (person_id) REFERENCES person(id)<br>    )</td>
    </tr>
</table>



Looking at the structure of the 'interview' table we can see that 'person_id' column in 'interview' references 'id' columns from 'person', which means that we can use those columns as common identifiers of the two tables.


```python
%sql SELECT * \
FROM interview \
JOIN person \
ON interview.person_id = person.id \
WHERE interview.person_id = 16371 OR interview.person_id = 14887
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>person_id</th>
        <th>transcript</th>
        <th>id</th>
        <th>name</th>
        <th>license_id</th>
        <th>address_number</th>
        <th>address_street_name</th>
        <th>ssn</th>
    </tr>
    <tr>
        <td>14887</td>
        <td>I heard a gunshot and then saw a man run out. He had a &quot;Get Fit Now Gym&quot; bag. The membership number on the bag started with &quot;48Z&quot;. Only gold members have those bags. The man got into a car with a plate that included &quot;H42W&quot;.</td>
        <td>14887</td>
        <td>Morty Schapiro</td>
        <td>118009</td>
        <td>4919</td>
        <td>Northwestern Dr</td>
        <td>111564949</td>
    </tr>
    <tr>
        <td>16371</td>
        <td>I saw the murder happen, and I recognized the killer from my gym when I was working out last week on January the 9th.</td>
        <td>16371</td>
        <td>Annabel Miller</td>
        <td>490173</td>
        <td>103</td>
        <td>Franklin Ave</td>
        <td>318771143</td>
    </tr>
</table>



By simply using the ids of Morty and Annabel we can find their interview transcripts detailing their witness reports. Both transcripts point us to the "Get Fit Now" gym.

## Get Fit Now Members


```python
%sql SELECT sql \
FROM sqlite_master \
WHERE name = 'get_fit_now_member'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>sql</th>
    </tr>
    <tr>
        <td>CREATE TABLE get_fit_now_member (<br>        id text PRIMARY KEY,<br>        person_id integer,<br>        name text,<br>        membership_start_date integer,<br>        membership_status text,<br>        FOREIGN KEY (person_id) REFERENCES person(id)<br>    )</td>
    </tr>
</table>




```python
%sql SELECT * \
FROM get_fit_now_member \
WHERE lower(membership_status) = 'gold' \
AND id LIKE '48Z%'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>id</th>
        <th>person_id</th>
        <th>name</th>
        <th>membership_start_date</th>
        <th>membership_status</th>
    </tr>
    <tr>
        <td>48Z7A</td>
        <td>28819</td>
        <td>Joe Germuska</td>
        <td>20160305</td>
        <td>gold</td>
    </tr>
    <tr>
        <td>48Z55</td>
        <td>67318</td>
        <td>Jeremy Bowers</td>
        <td>20160101</td>
        <td>gold</td>
    </tr>
</table>



Running the query on the table containing gym membership information we find two people matching the conditions of having a 'gold' membership status and membership id starting with '48Z' from Morty's witness report.

## Get Fit Now Checkin


```python
%sql SELECT sql \
FROM sqlite_master \
WHERE name = 'get_fit_now_check_in'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>sql</th>
    </tr>
    <tr>
        <td>CREATE TABLE get_fit_now_check_in (<br>        membership_id text,<br>        check_in_date integer,<br>        check_in_time integer,<br>        check_out_time integer,<br>        FOREIGN KEY (membership_id) REFERENCES get_fit_now_member(id)<br>    )</td>
    </tr>
</table>




```python
%sql SELECT * \
FROM get_fit_now_check_in check_in \
JOIN get_fit_now_member member \
ON check_in.membership_id = member.id \
WHERE person_id = 16371
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>membership_id</th>
        <th>check_in_date</th>
        <th>check_in_time</th>
        <th>check_out_time</th>
        <th>id</th>
        <th>person_id</th>
        <th>name</th>
        <th>membership_start_date</th>
        <th>membership_status</th>
    </tr>
    <tr>
        <td>90081</td>
        <td>20180109</td>
        <td>1600</td>
        <td>1700</td>
        <td>90081</td>
        <td>16371</td>
        <td>Annabel Miller</td>
        <td>20160208</td>
        <td>gold</td>
    </tr>
</table>



Verifying Annabel's witness report we find out that on 9 January she was in the gym from 1600 to 1700, indicating that the murderer was in the gym during that period of time as well.


```python
%sql SELECT * \
FROM get_fit_now_check_in \
WHERE check_in_date = 20180109 \
AND membership_id LIKE '48Z%'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>membership_id</th>
        <th>check_in_date</th>
        <th>check_in_time</th>
        <th>check_out_time</th>
    </tr>
    <tr>
        <td>48Z7A</td>
        <td>20180109</td>
        <td>1600</td>
        <td>1730</td>
    </tr>
    <tr>
        <td>48Z55</td>
        <td>20180109</td>
        <td>1530</td>
        <td>1700</td>
    </tr>
</table>



Querying the table for members who checked in to the gym on 9 January and have membership ids starting with '48Z' results in the same two people from before. Additionally, both of them we in the gym when Annabel was present so it is not possible to identify the killer.

## Driver's License
Fortunately, Morty's witness report details the killer entering a vehicle with car plate number containing "H42W".


```python
%sql SELECT sql \
FROM sqlite_master \
WHERE name = 'drivers_license'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>sql</th>
    </tr>
    <tr>
        <td>CREATE TABLE drivers_license (<br>        id integer PRIMARY KEY,<br>        age integer,<br>        height integer,<br>        eye_color text,<br>        hair_color text,<br>        gender text,<br>        plate_number text,<br>        car_make text,<br>        car_model text<br>    )</td>
    </tr>
</table>




```python
%sql SELECT * \
FROM drivers_license \
JOIN person \
ON drivers_license.id = person.license_id \
WHERE person.id = 28819 OR person.id = 67318
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>id</th>
        <th>age</th>
        <th>height</th>
        <th>eye_color</th>
        <th>hair_color</th>
        <th>gender</th>
        <th>plate_number</th>
        <th>car_make</th>
        <th>car_model</th>
        <th>id_1</th>
        <th>name</th>
        <th>license_id</th>
        <th>address_number</th>
        <th>address_street_name</th>
        <th>ssn</th>
    </tr>
    <tr>
        <td>423327</td>
        <td>30</td>
        <td>70</td>
        <td>brown</td>
        <td>brown</td>
        <td>male</td>
        <td>0H42W2</td>
        <td>Chevrolet</td>
        <td>Spark LS</td>
        <td>67318</td>
        <td>Jeremy Bowers</td>
        <td>423327</td>
        <td>530</td>
        <td>Washington Pl, Apt 3A</td>
        <td>871539279</td>
    </tr>
</table>



Querying the driver's license table with ids of the two suspects returns only one result from Jeremy Bowers with car plate number matching the description, suggesting that he was the killer who entered his own registered vehicle after the murder.

## Verify Answer


```python
%sql INSERT INTO solution VALUES (1, 'Jeremy Bowers');
%sql SELECT value FROM solution;
```

     * sqlite:///sql-murder-mystery.db
    1 rows affected.
     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>value</th>
    </tr>
    <tr>
        <td>Congrats, you found the murderer! But wait, there&#x27;s more... If you think you&#x27;re up for a challenge, try querying the interview transcript of the murderer to find the real villain behind this crime. If you feel especially confident in your SQL skills, try to complete this final step with no more than 2 queries. Use this same INSERT statement with your new suspect to check your answer.</td>
    </tr>
</table>



It seems that we have indeed identified the correct killer! Now a bonus challenge requires us to identify the mastermind behind this murder.


```python
%sql SELECT * \
FROM interview \
WHERE person_id = 67318
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>person_id</th>
        <th>transcript</th>
    </tr>
    <tr>
        <td>67318</td>
        <td>I was hired by a woman with a lot of money. I don&#x27;t know her name but I know she&#x27;s around 5&#x27;5&quot; (65&quot;) or 5&#x27;7&quot; (67&quot;). She has red hair and she drives a Tesla Model S. I know that she attended the SQL Symphony Concert 3 times in December 2017.<br></td>
    </tr>
</table>



Looking into the interview transcript of the murder we gather some identifying features of this mastermind.


```python
%sql SELECT * \
FROM person \
JOIN drivers_license dl \
ON person.license_id = dl.id \
JOIN facebook_event_checkin fb \
ON fb.person_id = person.id \
JOIN income \
ON income.ssn = person.ssn \
WHERE height BETWEEN 65 AND 67 \
AND lower(hair_color) = 'red' \
AND car_make = 'Tesla' \
AND car_model = 'Model S'
```

     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>id</th>
        <th>name</th>
        <th>license_id</th>
        <th>address_number</th>
        <th>address_street_name</th>
        <th>ssn</th>
        <th>id_1</th>
        <th>age</th>
        <th>height</th>
        <th>eye_color</th>
        <th>hair_color</th>
        <th>gender</th>
        <th>plate_number</th>
        <th>car_make</th>
        <th>car_model</th>
        <th>person_id</th>
        <th>event_id</th>
        <th>event_name</th>
        <th>date</th>
        <th>ssn_1</th>
        <th>annual_income</th>
    </tr>
    <tr>
        <td>99716</td>
        <td>Miranda Priestly</td>
        <td>202298</td>
        <td>1883</td>
        <td>Golden Ave</td>
        <td>987756388</td>
        <td>202298</td>
        <td>68</td>
        <td>66</td>
        <td>green</td>
        <td>red</td>
        <td>female</td>
        <td>500123</td>
        <td>Tesla</td>
        <td>Model S</td>
        <td>99716</td>
        <td>1143</td>
        <td>SQL Symphony Concert</td>
        <td>20171206</td>
        <td>987756388</td>
        <td>310000</td>
    </tr>
    <tr>
        <td>99716</td>
        <td>Miranda Priestly</td>
        <td>202298</td>
        <td>1883</td>
        <td>Golden Ave</td>
        <td>987756388</td>
        <td>202298</td>
        <td>68</td>
        <td>66</td>
        <td>green</td>
        <td>red</td>
        <td>female</td>
        <td>500123</td>
        <td>Tesla</td>
        <td>Model S</td>
        <td>99716</td>
        <td>1143</td>
        <td>SQL Symphony Concert</td>
        <td>20171212</td>
        <td>987756388</td>
        <td>310000</td>
    </tr>
    <tr>
        <td>99716</td>
        <td>Miranda Priestly</td>
        <td>202298</td>
        <td>1883</td>
        <td>Golden Ave</td>
        <td>987756388</td>
        <td>202298</td>
        <td>68</td>
        <td>66</td>
        <td>green</td>
        <td>red</td>
        <td>female</td>
        <td>500123</td>
        <td>Tesla</td>
        <td>Model S</td>
        <td>99716</td>
        <td>1143</td>
        <td>SQL Symphony Concert</td>
        <td>20171229</td>
        <td>987756388</td>
        <td>310000</td>
    </tr>
</table>



Simply by using the height, hair color, and car make/model we were able to narrow the search down to only one suspect Miranda Priestly. Additionally, from the table we observe that she has indeed attended the SQL Symphony Concert thrice in December 2017 and makes over 300K annually, matching the murderer's interview transcript.


```python
%sql INSERT INTO solution VALUES (1, 'Miranda Priestly');
%sql SELECT value FROM solution;
```

     * sqlite:///sql-murder-mystery.db
    1 rows affected.
     * sqlite:///sql-murder-mystery.db
    Done.
    




<table>
    <tr>
        <th>value</th>
    </tr>
    <tr>
        <td>Congrats, you found the brains behind the murder! Everyone in SQL City hails you as the greatest SQL detective of all time. Time to break out the champagne!</td>
    </tr>
</table>



And as suspected, Miranda is the mastermind behind this murder!
