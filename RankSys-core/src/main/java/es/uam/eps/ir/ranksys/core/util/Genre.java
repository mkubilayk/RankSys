package es.uam.eps.ir.ranksys.core.util;

public enum Genre {
	Action("Action"),
	Adventure("Adventure"),
	Animation("Animation"),
	Childrens("Children's"),
	Comedy("Comedy"),
	Crime("Crime"),
	Documentary("Documentary"),
	Drama("Drama"),
	Fantasy("Fantasy"),
	FilmNoir("Film-Noir"),
	Horror("Horror"),
	Musical("Musical"),
	Mystery("Mystery"),
	Romance("Romance"),
	SciFi("Sci-Fi"),
	Thriller("Thriller"),
	War("War"),
	Western("Western");


    private final String name;       

    private Genre(String s) {
        name = s;
    }

    public boolean equals(String otherName){
        return (otherName == null)? false:name.equals(otherName);
    }

    public String toString(){
       return name;
    }

}